# %%
import sqlite3
import re
import json
import torch
import pandas as pd

from transformers import AutoTokenizer, AutoModelForCausalLM
from openai import OpenAI


# %%
class TabularQAMiddleware:
    def __init__(self, llm_type="hf", model_name=None, device="auto", api_key=None):
        """
        llm_type: "hf" (Hugging Face) or "openai"
        model_name: Hugging Face 모델 이름 or OpenAI 모델명 (예: "gpt-4o")
        device: "auto" or "cuda"/"cpu"
        api_key: OpenAI API Key (llm_type="openai"일 경우 필요)
        """
        self.llm_type = llm_type
        self.model_name = model_name

        api_key = ""
        if llm_type == "hf":
            if AutoTokenizer is None:
                raise ImportError("Hugging Face Transformers not installed.")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if device == "auto" else None
            )
        elif llm_type == "openai":
            if api_key is None:
                raise ValueError("OpenAI API key is required for llm_type='openai'")
            self.client = OpenAI(api_key=api_key)
        else:
            raise ValueError("llm_type must be 'hf' or 'openai'")

    # -------------------------
    # 0. LLM generate 함수
    # -------------------------
    def llm_generate(self, prompt, max_new_tokens=512, temperature=0.1):
        """
        llm_type에 따라 LLM에서 텍스트 생성
        """
        if self.llm_type == "hf":
            # Hugging Face 모델
            message = [{"role": "user", "content": prompt}]
            text = self.tokenizer.apply_chat_template(
                message, tokenize=False, add_generation_prompt=True
            )
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=0.8,
                do_sample=True
            )
            output = self.tokenizer.decode(
                generated_ids[0][model_inputs.input_ids.size(1):],
                skip_special_tokens=True
            )
            return output.strip()

        elif self.llm_type == "openai":
            # OpenAI Chat Completions (SDK 1.x)
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_p=0.8
            )
            return response.choices[0].message.content.strip()

        else:
            raise ValueError("Unsupported llm_type")
    # -------------------------
    # 1. WikiTable → SQLite 변환
    # -------------------------
    def sanitize_column_names(self, headers):
        seen = {}
        clean_headers = []
        for h in headers:
            name = re.sub(r"[^\w]", "_", h.strip())
            if name in seen:
                seen[name] += 1
                name = f"{name}_{seen[name]}"
            else:
                seen[name] = 1
            clean_headers.append(name)
        return clean_headers

    def infer_column_types(self, rows):
        cols = list(zip(*rows))
        inferred_types = []
        for values in cols:
            col_type = "INTEGER"
            for val in values:
                val = str(val).replace(",", "").strip()
                if re.match(r"^\d+$", val):
                    continue
                try:
                    float(val)
                    col_type = "REAL"
                except:
                    col_type = "TEXT"
                    break
            inferred_types.append(col_type)
        return inferred_types

    def wikitable_to_sqlite(self, table_json, db_path=":memory:"):
        """WikiTable JSON → SQLite DB 변환"""
        table_info = table_json["table"]
        raw_headers = table_info["header"]
        headers = self.sanitize_column_names(raw_headers)
        rows = table_info["rows"]
        table_name = table_info.get("name", "wikitable").replace("/", "_").replace("-", "_").replace(".tsv", "")

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        types = self.infer_column_types(rows)
        escaped_headers = [f'"{h}" {t}' for h, t in zip(headers, types)]
        create_sql = f'CREATE TABLE {table_name} ({", ".join(escaped_headers)});'
        cursor.execute(create_sql)

        placeholders = ", ".join(["?"] * len(headers))
        insert_sql = f'INSERT INTO {table_name} VALUES ({placeholders});'
        cursor.executemany(insert_sql, rows)
        conn.commit()

        return conn, table_name, create_sql

    # -------------------------
    # 2. SQL 스텝 생성
    # -------------------------
    def generate_sql_steps(self, question, schema, table_preview):
        prompt = f"""
You are an expert in SQLite.
Decompose the user question into reasoning steps and generate an SQL for each step.

[User Question]
{question}

[Database Schema]
{schema}

[Table Preview]
{table_preview}

Return answer in JSON array format:
[
  {{"step": 1, "sql": "SELECT ..."}},
  {{"step": 2, "sql": "..."}}
]
"""
        response = self.llm_generate(prompt)
        json_like = re.findall(r"\[.*\]", response, re.DOTALL)
        return json_like[0] if json_like else response

    # -------------------------
    # 3. SQL 실행 + Self-healing (이전 시도 기록)
    # -------------------------
    def clean_sql_output(self, sql_text: str) -> str:
        """
        Remove markdown fences, extra explanation, and keep only the SQL query.
        """
        # Remove markdown code block markers
        cleaned = sql_text.strip()
        cleaned = re.sub(r"^```[a-zA-Z]*", "", cleaned)  # 시작 부분 ```sql 같은거 제거
        cleaned = re.sub(r"```$", "", cleaned)           # 끝 부분 ``` 제거
        cleaned = cleaned.strip()
        
        # SQL 끝나는 세미콜론 이후 불필요한 설명 제거
        if ";" in cleaned:
            first_semicolon_idx = cleaned.find(";")
            cleaned = cleaned[:first_semicolon_idx+1]
        
        return cleaned

    def execute_sql_with_retry(self, conn, original_sql, step_idx=1, schema=None, max_retries=5):
        sql = self.clean_sql_output(original_sql)
        cursor = conn.cursor()
        history = []

        for attempt in range(1, max_retries+1):
            try:
                cursor.execute(sql)
                return sql, cursor.fetchall()  # 성공 시 최종 SQL과 결과 반환
            except sqlite3.Error as e:
                error_msg = str(e)
                history.append((sql, error_msg))

                # 이전 시도 기록 + 스키마 + 에러 메시지 피드백
                conversation = f"You are an expert in SQLite.\n"
                conversation += f"We are fixing SQL for step {step_idx}.\n\n"
                if schema:
                    conversation += f"Database Schema:\n{schema}\n\n"

                for i, (prev_sql, prev_error) in enumerate(history, 1):
                    conversation += f"Attempt {i}:\nSQL: {prev_sql}\nError: {prev_error}\n\n"

                conversation += (
                    f"Attempt {len(history)+1}: "
                    "Generate a corrected SQL that avoids all previous errors.\n"
                    "Only output the corrected SQL without markdown fences or explanation."
                )

                # 새 SQL 생성 및 클린 처리
                sql = self.llm_generate(conversation)
                sql = self.clean_sql_output(sql)

                print(f"[Step {step_idx}] Attempt {attempt} failed with error: {error_msg}")
                print(f"New SQL candidate: {sql}\n")

        # 5회 시도 후 실패
        return sql, None



    # -------------------------
    # 4. Tabular QA 최종 답변
    # -------------------------
    def tabular_qa_with_sql(self, question, schema, table_preview, sql_steps):
        prompt = f"""
You are a Table Question Answering expert.

We have the following information:
- User Question: {question}
- Table Schema: {schema}
- Table Preview: {table_preview}
- SQL Steps and Results: {sql_steps}

Using all the information above, generate the FINAL ANSWER.
Output only the final answer. Do not include any explanation or reasoning.
"""
        return self.llm_generate(prompt)

    # -------------------------
    # 5. 전체 파이프라인
    # -------------------------
    def run(self, question, schema, table_preview, conn):
        sql_steps_str = self.generate_sql_steps(question, schema, table_preview)

        results = []
        try:
            sql_steps = json.loads(sql_steps_str)
        except json.JSONDecodeError:
            sql_steps = [{"step": 1, "sql": sql_steps_str}]

        for step in sql_steps:
            step_idx = step.get("step", 0)
            sql = step.get("sql")
            final_sql, result = self.execute_sql_with_retry(conn, sql, step_idx)

            results.append({
                "step": step_idx,
                "sql": final_sql,
                "result": result,
                "error": None if result is not None else "Execution failed after 5 retries"
            })

        final_answer = self.tabular_qa_with_sql(question, schema, table_preview, results)
        return final_answer, results

    # -------------------------
    # 6. WikiTable 샘플 실행
    # -------------------------
    def run_on_wikitable(self, sample, preview_rows=5):
        conn, table_name, create_sql = self.wikitable_to_sqlite(sample)
        question = sample["question"]

        headers = sample["table"]["header"]
        rows = sample["table"]["rows"][:preview_rows]
        table_preview = pd.DataFrame(rows, columns=headers).to_string(index=False)

        answer, steps = self.run(question, create_sql, table_preview, conn)
        return answer, steps


# %%
from datasets import load_dataset

# %%
# 샘플 데이터 로드
wikitablequestions = load_dataset("wikitablequestions", trust_remote_code=True)

# %%
sample = wikitablequestions['test'][136]
sample

# %%
middleware = TabularQAMiddleware(llm_type="openai", model_name="gpt-4o")
answer, steps = middleware.run_on_wikitable(sample)

print("SQL Steps + Results:", steps)
print("Final Answer:", answer)

# %%

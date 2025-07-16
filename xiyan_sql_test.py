# %%
import re
# import torch
import sqlite3
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# %%
model_name = "XGenerationLab/XiYanSQL-QwenCoder-3B-2502"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# %%
def generate_sql_response(prompt):
    message = [{'role': 'user', 'content': prompt}]
    text = tokenizer.apply_chat_template(
        message,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    generated_ids = model.generate(
        **model_inputs,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=1024,
        temperature=0.1,
        top_p=0.8,
        do_sample=True,
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

# %%
nl2sqlite_template_cn = """你是一名{dialect}专家，现在需要阅读并理解下面的【数据库schema】描述，以及可能用到的【参考信息】，并运用{dialect}知识生成sql语句回答【用户问题】。
【用户问题】
{question}
【数据库schema】
{db_schema}
【参考信息】
{evidence}
【用户问题】
{question}
```sql"""

nl2sqlite_template_en = """You are an expert in {dialect}.
Please read and understand the following [Database Schema] description and the [Reference Information] (if provided),
and then use your knowledge of {dialect} to generate an SQL query that answers the [User Question].
[User Question]
{question}
[Database Schema]
{db_schema}
[Reference Information]
{evidence}
[User Question]
{question}
```sql"""

to_be1 = """You are an expert in {dialect}.
Please read and understand the following [Database Schema], [Table Details],
and the [Reference Information] (if provided), all of which are provided for reference.
Then, use your knowledge of {dialect} to generate an SQL query that answers the [User Question].
[User Question]
{question}
[Database Schema]
{db_schema}
[Table Details]
{table_info}
[Reference Information]
{evidence}
[User Question]
{question}
```sql"""

to_be2 = """You are a Table Question Answering expert.
Given the [User Question], and the provided [Table] and [Reference Information] (if any),
directly generate the correct answer in natural language.
[User Question]
{question}
[Table]
{table}
[Reference Information]
{evidence}
[User Question]
{question}
"""


## dialects -> ['SQLite', 'PostgreSQL', 'MySQL']
prompt = nl2sqlite_template_cn.format(dialect="", db_schema="", question="", evidence="")
prompt = nl2sqlite_template_cn.format(
    dialect="sqlite",
    db_schema="CREATE TABLE olympics (Name TEXT, Country TEXT, Medals INT);",
    question="Who won the most medals?",
    evidence="Table contains medal counts for Olympic athletes."
)
generate_sql_response(prompt)

# %%
def infer_column_types(rows):
    cols = list(zip(*rows))
    inferred_types = []
    for values in cols:
        col_type = "INTEGER"
        for val in values:
            val = val.replace(",", "").strip()
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

def sanitize_column_names(headers):
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

def wikitable_to_sqlite(table_json, db_path=":memory:"):
    table_info = table_json["table"]
    raw_headers = table_info["header"]
    headers = sanitize_column_names(raw_headers)

    rows = table_info["rows"]
    table_name = table_info.get("name", "wikitable").replace("/", "_").replace("-", "_").replace(".tsv", "")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    types = infer_column_types(rows)
    escaped_headers = [f'"{h}" {t}' for h, t in zip(headers, types)]
    create_sql = f'CREATE TABLE {table_name} ({", ".join(escaped_headers)});'
    cursor.execute(create_sql)

    placeholders = ", ".join(["?"] * len(headers))
    insert_sql = f'INSERT INTO {table_name} VALUES ({placeholders});'
    cursor.executemany(insert_sql, rows)
    conn.commit()

    return conn, table_name, create_sql

def generate_prompt(sample, create_sql):
    question = sample["question"]
    evidence = f'Table contains columns: {", ".join(sample["table"]["header"])}. The question asks: "{question}"'
    return nl2sqlite_template_cn.format(
        dialect="sqlite",
        db_schema=create_sql,
        question=question,
        evidence=evidence
    )

def execute_sql_and_fetch(conn, sql):
    try:
        cursor = conn.cursor()
        cursor.execute(sql)
        result = cursor.fetchall()
        return result
    except sqlite3.Error as e:
        print(f"SQL 실행 오류: {e}")
        return None

def debug_db(conn, table_name):
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    print("생성된 테이블 목록:", [t[0] for t in tables])

    cursor.execute(f"PRAGMA table_info({table_name});")
    schema_info = cursor.fetchall()
    print("테이블 스키마:")
    for col in schema_info:
        # col = (cid, name, type, notnull, dflt_value, pk)
        print(f" - {col[1]} ({col[2]})")

    cursor.execute(f"SELECT * FROM {table_name} LIMIT 5;")
    rows = cursor.fetchall()
    print("테이블 내용 미리보기 (최대 5개):")
    for row in rows:
        print(row)

def evaluate_table_qa_sample(sample, debug=False):
    headers = sample["table"]["header"]
    rows = sample["table"]["rows"]
    df = pd.DataFrame(rows, columns=headers)
    conn, table_name, create_sql = wikitable_to_sqlite(sample)
    prompt = generate_prompt(sample, create_sql)
    response = generate_sql_response(prompt)
    result = execute_sql_and_fetch(conn, response)
    
    if debug:
        print(sample['question'], sample['answers'])
        print(df.head())
        print(conn, table_name, create_sql)
        print(prompt)
        print(response)
        print(result)
    return result

def is_correct_answer(sql_result, gold_answers):
    if not sql_result or sql_result[0][0] is None:
        return False
    pred = str(sql_result[0][0]).strip().lower()
    gold_norm = [str(ans).strip().lower() for ans in gold_answers]
    return pred in gold_norm
# %%
wikitablequestions = load_dataset("wikitablequestions", trust_remote_code=True)

# %%
sample = wikitablequestions['train'][0]
result = evaluate_table_qa_sample(sample, debug=True)
result

# %%
is_correct_answer(result, sample['answers'])

# %%
label_txt_path = "question_difficulty_result.txt"

label_dict = {}
with open(label_txt_path, "r", encoding="utf-8") as f:
    for line in f:
        idx, label = line.strip().split()
        label_dict[int(idx)] = int(label)


test_data = wikitablequestions['test']

def add_label(example, idx):
    return {"label": label_dict.get(idx, -1)}

# Dataset.map으로 label 컬럼 추가 (with_indices=True를 꼭 지정)
test_data = test_data.map(add_label, with_indices=True)
test_data

# %%
label_1_data = test_data.filter(lambda x: x["label"] == 1)
label_1_data

# %%
ans_cnt = 0
for i, sample in enumerate(test_data):
    print(i, end=' ')
    result = evaluate_table_qa_sample(sample)
    ans = is_correct_answer(result, sample['answers'])
    print(ans)
    ans_cnt += ans

# %%
ans_cnt

# %%

model_size = '32b'
result_txt_path = f"xiyan_sql_test_result_{model_size}.txt"

result_dict = {}
with open(result_txt_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

i = 0
while i < len(lines):
    line = lines[i].strip()

    # 정상 줄: "0 True" 또는 "1 False"
    if line and line[0].isdigit() and ("True" in line or "False" in line):
        idx, value = line.split()
        result_dict[int(idx)] = value == "True"
        i += 1

    # 에러 + 다음 줄에 결과
    elif line and line[0].isdigit() and "SQL 실행 오류" in line:
        idx = int(line.split()[0])
        i += 1
        if i < len(lines):
            value = lines[i].strip()
            result_dict[idx] = value == "True"
        i += 1

    else:
        i += 1  # 혹시 모를 빈 줄/예외 대응

# %%
def add_pass_fail(example, idx):
    return {f"passed_{model_size}": result_dict.get(idx, None)}  # None은 결과 없음

test_data = test_data.map(add_pass_fail, with_indices=True)

# %%
df = test_data.to_pandas()
summary = df.groupby("label")[f"passed_{model_size}"].agg(
    total="count",
    correct="sum"
).reset_index()
summary["accuracy"] = summary["correct"] / summary["total"]
summary

# %%
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False

plt.figure(figsize=(8, 4))
plt.bar(summary["label"], summary["accuracy"])
plt.title(f"Label별 정확도 ({model_size.upper()} 모델)")
plt.xlabel("Label")
plt.ylabel("정답 비율")
plt.xticks(summary["label"])
plt.ylim(0, 1)
plt.show()

# %%
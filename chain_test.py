# %%
import pandas as pd
from openai import OpenAI
from datasets import load_dataset
dataset = load_dataset("wikitablequestions", trust_remote_code=True)

def generate_chain_code_from_gpt_v1(question, headers, rows, api_key, model="gpt-3.5-turbo"):
    client = OpenAI(api_key=api_key)

    system_prompt = """
You are an assistant that converts natural language questions about a tabular dataset into a chain of pandas operations using only 'df'.
Rules:
- Use only 'df', and apply chaining rules. No intermediate variables.
- Output must be a complete pandas expression like: df = df[...] ... df
- Assume 'df' is already defined as a pandas DataFrame with proper column headers.
- Output only code. No explanation.
"""

    user_prompt = f"""
Here are some examples:

---

Question: What was the last year where this team was a part of the USL A-League?
Table headers: ['Year', 'Division', 'League', 'Regular Season', 'Playoffs', 'Open Cup', 'Avg. Attendance']
Table rows: [...]
Answer Code:
df = df[df["League"] == "USL A-League"]
df = df["Year"]
df = df.max()
df

---

Question: In what city did Piotr's last 1st place finish occur?
Table headers: ['Year', 'Competition', 'Venue', 'Position', 'Event', 'Notes']
Table rows: [...]
Answer Code:
df = df[df["Position"] == "1st"]
df = df.sort_values("Year")
df = df.iloc[-1]
df = df["Venue"]
df

---

Now answer the following:

Question: {question}
Table headers: {headers}
Table rows: {rows}
Answer Code:
"""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0
    )

    return response.choices[0].message.content.strip()

# 실행 루프
correct = 0
total = 10

for i in range(10, 20):
    sample = dataset["train"][i]
    headers = sample["table"]["header"]
    rows = sample["table"]["rows"]
    question = sample["question"]
    answers = sample["answers"]

    print(f"\n{'='*60}")
    print(f"📌 Sample #{i}")
    print(f"❓ Question: {question}")
    print(f"📋 Table (first 3 rows):")
    df_preview = pd.DataFrame(rows, columns=headers).head(3)
    print(df_preview)

    try:
        # GPT 호출
        code = generate_chain_code_from_gpt_v1(question, headers, rows, api_key="api_key")

        print("\n💬 Generated Code:\n", code)

        # 실행
        df = pd.DataFrame(rows, columns=headers)
        exec_locals = {"df": df.copy()}
        exec(code, {}, exec_locals)
        result = exec_locals["df"]

        # 결과 비교
        result_str = str(result)
        is_correct = result_str in answers
        if is_correct:
            correct += 1

        print(f"✅ Predicted: {result_str}")
        print(f"🎯 Ground Truth: {answers}")
        print(f"🧾 Result: {'✅ Correct' if is_correct else '❌ Wrong'}")

    except Exception as e:
        print("❌ Execution Error:", e)

print(f"\n{'='*60}")
print(f"🔚 Evaluation Summary: {correct}/{total} correct → Accuracy: {correct/total:.1%}")

# %%

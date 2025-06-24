# %%
import random
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

# %%
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
def one_sample(i):
    sample = dataset["train"][i]
    headers = sample["table"]["header"]
    rows = sample["table"]["rows"]
    question = sample["question"]
    answers = sample["answers"]

    print(f"Question: {question}")
    print(f"Ground Truth: {answers}")
    df_origin = pd.DataFrame(rows, columns=headers)
    print(df_origin.head())
    return df_origin

def pick_sorted_unique_numbers(n, m, seed=42):
    random.seed(seed)
    numbers = random.sample(range(n), m)
    return sorted(numbers)


# %%
samples = [409, 1679, 1824, 2286, 3657, 4012, 4506, 8935, 10476, 11087]
samples = [
    106, 409, 434, 488, 520, 711, 1424, 1519, 1535, 1584,
    1674, 1679, 1824, 2045, 2286, 2547, 2615, 3257, 3527, 3582,
    3611, 3657, 3811, 4012, 4333, 4506, 4552, 4557, 5514, 5574,
    5635, 5881, 6224, 6873, 6912, 6924, 7359, 7527, 8279, 8785,
    8928, 8935, 9195, 9654, 9674, 9863, 9891, 10476, 10647, 11087
]
pick_sorted_unique_numbers(len(dataset["train"]), 50)

# %% 520
"""
Question: what was the name of the mission previous to cosmos 300?
Ground Truth: ['Luna 15']
         Launch date Operator               Name Sample origin  \
0      June 14, 1969           Luna E-8-5 No.402      The Moon   
1      July 13, 1969                     Luna 15      The Moon   
2  23 September 1969                  Cosmos 300      The Moon   
3    22 October 1969                  Cosmos 305      The Moon   
4    6 February 1970           Luna E-8-5 No.405      The Moon   

  Samples returned Recovery date                        Mission result  
0             None             -               Failure\nLaunch failure  
1             None             -     Failure\nCrash-landed on the Moon  
2             None             -  Failure\nFailed to leave Earth orbit  
3             None             -  Failure\nFailed to leave Earth orbit  
4             None             -               Failure\nLaunch failure  

문제
1. score를 home/away team을 이해하고 parsing해야 함
   해당 사례에서는 home score away 컬럼순이었는데 반대라면?
"""

idx = 5
print(samples[idx])
df = one_sample(samples[idx]).copy()
df

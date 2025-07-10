# %%
import time
import pandas as pd
from openai import OpenAI
from datasets import load_dataset

# %%
client = OpenAI(api_key="")

# %%
def classify_tabular_question_with_gpt(question, df, max_retry=3):
    icl_prompt = """
The following are examples of classifying a question based on the level of reasoning required to answer it using a table.

(1: Answerable using only column names / 
 2: Requires exploring table data / 
 3: Requires external knowledge beyond the table)

[Example 1]
Question: Which team had the most wins?
Columns: ["Season", "Team", "Wins", "Losses"]
Rows: [["2019", "Lakers", "52", "30"], ["2020", "Heat", "44", "28"]]
Answer: 1
Reason: The presence of the column "Wins" is sufficient to understand how the question should be answered.

[Example 2]
Question: How many plants are in Algeria?
Columns: ["Plant Name", "Country", "Startup Date", "Capacity"]
Rows: [["Arzew", "Algeria", "1990", "7.8"], ["GL1Z", "Qatar", "1980", "5.2"]]
Answer: 2
Reason: The answer requires filtering rows where Country = "Algeria".

[Example 3]
Question: When was the first win by decision?
Columns: ["Ship", "Method", "Date"]
Rows: [["San Francisco", "Decision", "1944"], ["Detroit", "Sunk", "1945"]]
Answer: 2
Reason: We need to filter rows where Method includes "Decision" and then find the earliest date.

[Example 4]
Question: Which country has produced the most drivers?
Columns: ["Driver"]
Rows: [["David Brown"], ["Jacques Perónn"], ["Lucien Farnand"]]
Answer: 3
Reason: The country information is not present in the table and requires external knowledge.

---
"""

    # 🔧 df에서 headers, rows 추출
    headers = list(df.columns)
    sampled_rows = df.head(3).values.tolist()  # 최대 3행만

    # 프롬프트 구성
    header_str = str(headers)
    row_str = "[" + ", ".join(str(row) for row in sampled_rows) + "]"
    prompt = icl_prompt + f"""
Classify the following question.

Question: {question}
Columns: {header_str}
Rows: {row_str}
Answer:"""
    

    for attempt in range(max_retry):
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
            )
            content = response.choices[0].message.content.strip()
            for token in content.split():
                if token in {"1", "2", "3"}:
                    return int(token)
            return None
        except Exception as e:
            print(f"[Error {attempt+1}] {e}")
            time.sleep(1)
    return None

# %%
wikitablequestions = load_dataset("wikitablequestions", trust_remote_code=True)

# %%
sample = wikitablequestions['test'][0]
question = sample['question']
df = pd.DataFrame(sample['table']['rows'], columns=sample['table']['header'])

label = classify_tabular_question_with_gpt(question, df)
print(f"질문: {question}")
print(f"테이블")
print(df.head())
print(f"→ GPT 분류 결과: {label}")

# %%
for idx in range(0, len(wikitablequestions['test'])):
    sample = wikitablequestions['test'][idx]
    question = sample['question']
    df = pd.DataFrame(sample['table']['rows'], columns=sample['table']['header'])
    label = classify_tabular_question_with_gpt(question, df)
    print(idx, label)

# %%
import matplotlib.pyplot as plt
from collections import Counter

# 1. 결과 파일 읽기
with open("question_difficulty_result.txt", "r") as f:
    lines = f.readlines()

# 2. 숫자만 추출
labels = [int(line.strip().split()[1]) for line in lines]

# 3. 개수 세기
counts = Counter(labels)
sorted_counts = [counts.get(i, 0) for i in [1, 2, 3]]

# 4. 시각화
plt.figure(figsize=(6, 4))
plt.bar(['1', '2', '3'], sorted_counts)
plt.title("Distribution of Question Types")
plt.xlabel("Category")
plt.ylabel("Count")
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.show()

# %%
total = sum(counts.values())

print("Category\tCount\tPercentage")
for label in [1, 2, 3]:
    count = counts.get(label, 0)
    percent = (count / total * 100) if total > 0 else 0
    print(f"{label}\t\t{count}\t{percent:.2f}%")
# %%

# %%
import json
import pandas as pd
from io import StringIO
from datasets import load_dataset


# %%
wikitablequestions = load_dataset("wikitablequestions", trust_remote_code=True)
wikitablequestions['train'][0]

# %%
sample = wikitablequestions['train'][0]
headers = sample["table"]["header"]
rows = sample["table"]["rows"]

df = pd.DataFrame(rows, columns=headers)
df.head()


# %% tab fact
tab_fact = load_dataset('tab_fact', 'tab_fact', trust_remote_code=True)
tab_fact['train'][0]

# %% 
df = pd.read_csv(StringIO(tab_fact['train'][0]['table_text']), sep="#")
df.head()


# %% fetaqa
fetaqa = load_dataset("DongfuJiang/FeTaQA", trust_remote_code=True)
fetaqa['train'][0]

# %%
columns = fetaqa['train'][0]['table_array'][0]
rows = fetaqa['train'][0]['table_array'][1:]

df = pd.DataFrame(rows, columns=columns)
df.head()


# %% mmqa
with open('../mmqa/Synthesized_three_table.json', 'r', encoding='utf-8') as f:
    mmqa_three_table = json.load(f)

# %%
mmqa_three_table[0]


# %%
spider = load_dataset("spider", trust_remote_code=True)

# %%
spider['train'][6]

# %%
wikisql = load_dataset("wikisql", trust_remote_code=True)
wikisql['train'][0]

# %%
columns = wikisql['train'][0]['table']['header']
rows = wikisql['train'][0]['table']['rows']

df = pd.DataFrame(rows, columns=columns)
df.head()

# %%
spider2 = load_dataset("xlangai/spider2-lite", trust_remote_code=True)
spider2

# %%
spider2['train'][0]
# %%

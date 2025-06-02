# %%
import pandas as pd
import random

# ---------------------------
# 유틸: 컬럼 선택 함수
# ---------------------------
def choose_column(df, col_type):
    if col_type == 'number':
        candidates = df.select_dtypes(include='number').columns.tolist()
    elif col_type == 'text':
        candidates = df.select_dtypes(include='object').columns.tolist()
    else:
        return None
    return random.choice(candidates) if candidates else None

# ---------------------------
# 템플릿: 자연어 설명 생성
# ---------------------------
TEMPLATES = {
    "filter_numeric": lambda col, cond, val: f"{col} 값이 {val:.1f}{cond}인 항목",
    "filter_text_match": lambda col, keyword: f"{col}에 '{keyword}'가 포함된 항목",
    "argmax": lambda col: f"{col} 값이 가장 큰 항목",
    "argmin": lambda col: f"{col} 값이 가장 작은 항목",
    "groupby_mean": lambda group_col, agg_col: f"{group_col} 별 {agg_col} 평균 계산"
}

# ---------------------------
# 상태 추적용 변수
# ---------------------------
used_filter_cols = set()
used_groupby_cols = set()

# ---------------------------
# Step 함수 정의
# ---------------------------
def apply_filter_numeric(df):
    col = choose_column(df, 'number')
    if col is None or col in used_filter_cols:
        return df, "(숫자형 컬럼 없음 또는 중복 필터 방지)", None, False, False

    values = sorted(df[col].dropna().unique())
    if len(values) < 4:
        return df, "(유효 필터 값 부족)", None, False, False

    val = random.uniform(values[1], values[-2])
    op = random.choice(['<', '<=', '>', '>=', '=='])

    if op == '<':
        filtered = df[df[col] < val]
    elif op == '<=':
        filtered = df[df[col] <= val]
    elif op == '>':
        filtered = df[df[col] > val]
    elif op == '>=':
        filtered = df[df[col] >= val]
    else:
        val = random.choice(values)
        filtered = df[df[col] == val]

    desc = TEMPLATES["filter_numeric"](col, op, val)
    used_filter_cols.add(col)
    return filtered, desc, None, True, False

def apply_filter_text_match(df):
    col = choose_column(df, 'text')
    if col is None or df[col].dropna().empty:
        return df, "(텍스트 컬럼 없음)", None, False, False

    all_text = " ".join(df[col].dropna().astype(str).tolist()).split()
    keywords = [word.strip(",.()") for word in all_text if len(word.strip(",.()")) >= 4]
    keywords = list(set(keywords))

    if not keywords:
        return df, "(키워드 없음)", None, False, False

    keyword = random.choice(keywords)
    filtered = df[df[col].str.contains(keyword, case=False, na=False)]
    desc = TEMPLATES["filter_text_match"](col, keyword)
    return filtered, desc, None, True, False

def apply_argmax(df):
    col = choose_column(df, 'number')
    if col is None or len(df) == 0:
        return df, "(argmax 불가)", None, False, False
    idx = df[col].idxmax()
    new_df = df.loc[[idx]]
    desc = TEMPLATES["argmax"](col)
    return new_df, desc, None, True, False

def apply_argmin(df):
    col = choose_column(df, 'number')
    if col is None or len(df) == 0:
        return df, "(argmin 불가)", None, False, False
    idx = df[col].idxmin()
    new_df = df.loc[[idx]]
    desc = TEMPLATES["argmin"](col)
    return new_df, desc, None, True, False

def apply_groupby_mean(df):
    group_col = choose_column(df, 'text')
    agg_col = choose_column(df, 'number')
    if group_col is None or agg_col is None or group_col in used_groupby_cols:
        return df, "(groupby 불가 또는 중복 방지)", None, False, False

    grouped = df.groupby(group_col)[agg_col].mean().reset_index().rename(columns={agg_col: f'{agg_col}_mean'})
    desc = TEMPLATES["groupby_mean"](group_col, agg_col)
    used_groupby_cols.add(group_col)
    return grouped, desc, None, True, False

# ---------------------------
# Step 함수 목록
# ---------------------------
STEP_FUNCTIONS = [
    apply_filter_numeric,
    apply_filter_text_match,
    apply_argmax,
    apply_argmin,
    apply_groupby_mean
]

# ---------------------------
# Chain 실행 함수
# ---------------------------
def run_reasoning_chain_relaxed(df, max_steps=5):
    global used_filter_cols, used_groupby_cols
    used_filter_cols = set()
    used_groupby_cols = set()

    current_df = df.copy()
    question_parts = []
    used_desc = set()
    step_log = [("초기 상태", current_df.copy())]

    for _ in range(max_steps):
        if len(current_df) > 3:
            candidates = STEP_FUNCTIONS
        else:
            candidates = [apply_argmax, apply_argmin]

        tried = set()
        while len(tried) < len(candidates):
            step_func = random.choice(candidates)
            if step_func in tried:
                continue
            tried.add(step_func)

            new_df, desc, _, changes_df, _ = step_func(current_df)

            if (desc in used_desc) or (changes_df and new_df.equals(current_df)):
                continue

            used_desc.add(desc)
            question_parts.append(desc)
            step_log.append((desc, new_df.copy()))
            current_df = new_df

            if len(current_df) <= 1 and len(step_log) >= 2:
                return question_parts, step_log, current_df
            break

    return question_parts, step_log, current_df

# ---------------------------
# 예시 데이터프레임 생성
# ---------------------------
entities = [f"Item{i}" for i in range(20)]
heights = random.choices(range(100, 200), k=20)
floors = random.choices(range(10, 60), k=20)
descriptions = random.choices(
    ["modern office", "first tower", "BP building", "commercial tower", "residential", "historic"],
    k=20
)

df_large_example = pd.DataFrame({
    "Entity": entities,
    "Height": heights,
    "Floors": floors,
    "Description": descriptions
})

# ---------------------------
# 체인 실행 및 출력
# ---------------------------
question_parts, step_log, final_df = run_reasoning_chain_relaxed(df_large_example, max_steps=5)

for i, (desc, df_state) in enumerate(step_log):
    print(f"[Step {i}] {desc}")
    display(df_state)

print("\n🧠 최종 질문 예시:")
print(" → ".join(question_parts))

print("\n📦 최종 df:")
display(final_df)

# %%


from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[2]
RAW_PATH = PROJECT_DIR / "data" / "raw" / "Lipophilicity.csv"
INTERIM_DIR = PROJECT_DIR / "data" / "interim"
AUDIT_DIR = PROJECT_DIR / "data" / "audit"
INTERIM_DIR.mkdir(parents=True, exist_ok=True)
AUDIT_DIR.mkdir(parents=True, exist_ok=True)

# 1. 读取原始数据
df = pd.read_csv(RAW_PATH)

original_count = len(df)

# 2. 将 exp 转换为数值
df["exp"] = pd.to_numeric(df["exp"], errors="coerce")

# 3. 记录被排除的数据及原因
excluded = []

for idx, row in df.iterrows():
    reasons = []

    if pd.isna(row["exp"]):
        reasons.append("exp缺失或无法转换为数字")

    if pd.isna(row["smiles"]) or not str(row["smiles"]).strip():
        reasons.append("SMILES缺失")

    if pd.notna(row["exp"]) and row["exp"] > 10:
        reasons.append("exp异常大于10")

    if reasons:
       item = row.to_dict()
       item["original_index"] = idx
       item["exclusion_reason"] = "; ".join(reasons)
       excluded.append(item)

# 4. 创建待清洗数据
clean = df.copy()

# 排除无效标签
clean = clean.dropna(subset=["exp"])

# 排除缺失SMILES
clean = clean[
    clean["smiles"].notna()
    & clean["smiles"].astype(str).str.strip().ne("")
]

# 排除初步识别出的异常标签
clean = clean[clean["exp"] <= 10].copy()

before_dedup = len(clean)

# 5. 删除完全重复的行
clean = clean.drop_duplicates().copy()

# 6. 保存清洗结果
clean.to_csv(INTERIM_DIR / "Lipophilicity_clean.csv", index=False)

# 保存排除记录
if excluded:
    excluded_df = pd.DataFrame(excluded)
    excluded_df.to_csv(AUDIT_DIR / "excluded_records.csv", index=False)

# 7. 输出清洗报告
print("===== 数据清洗报告 =====")
print("原始记录数：", original_count)
print("排除无效记录数：", len(excluded))
print("完全重复记录删除数：", before_dedup - len(clean))
print("清洗后记录数：", len(clean))

print("\n清洗后 exp 统计：")
print(clean["exp"].describe())

print("\n已生成文件：")
print("Lipophilicity_clean.csv")
print("excluded_records.csv")

print("\n数据清洗完成！")

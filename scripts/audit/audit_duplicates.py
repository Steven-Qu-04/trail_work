
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[2]
INTERIM_DIR = PROJECT_DIR / "data" / "interim"
AUDIT_DIR = PROJECT_DIR / "data" / "audit"
AUDIT_DIR.mkdir(parents=True, exist_ok=True)

# 读取最终有效数据
df = pd.read_csv(INTERIM_DIR / "Lipophilicity_valid.csv")

# 按完全相同的 SMILES 分组
groups = df.groupby("smiles").agg(
    record_count=("exp", "size"),
    exp_unique_count=("exp", "nunique"),
    exp_values=("exp", lambda x: list(x.unique())),
    compound_ids=("CMPD_CHEMBLID", lambda x: list(x))
)

# 找出重复 SMILES
duplicates = groups[groups["record_count"] > 1].copy()

# 找出同一个 SMILES 对应不同 exp 的情况
conflicts = duplicates[
    duplicates["exp_unique_count"] > 1
].copy()

# 保存审计结果
duplicates.to_csv(AUDIT_DIR / "duplicate_smiles.csv")
conflicts.to_csv(AUDIT_DIR / "conflicting_labels.csv")

print("===== 重复分子审计 =====")
print("有效数据记录数：", len(df))
print("不同 SMILES 数量：", df["smiles"].nunique())
print("重复 SMILES 组数：", len(duplicates))
print("标签不一致的 SMILES 组数：", len(conflicts))

print("\n标签不一致的记录示例：")
print(conflicts.head(10).to_string())

print("\n已生成：")
print("duplicate_smiles.csv")
print("conflicting_labels.csv")

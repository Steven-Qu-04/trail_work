
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[2]

# 读取原始数据
df = pd.read_csv(PROJECT_DIR / "data" / "raw" / "Lipophilicity.csv")

print("===== 1. 缺失数据 =====")
print(df[df.isnull().any(axis=1)])

print("\n===== 2. 完全重复的行 =====")
print("完全重复行数量：", df.duplicated().sum())

print("\n===== 3. 重复的分子编号 =====")
print("重复编号数量：", df.duplicated(
    subset=["CMPD_CHEMBLID"]
).sum())

print("\n===== 4. 重复的 SMILES =====")
print("重复 SMILES 数量：", df.duplicated(
    subset=["smiles"]
).sum())

print("\n===== 5. exp 转换为数值的检查 =====")
exp_numeric = pd.to_numeric(df["exp"], errors="coerce")

print("原始 exp 缺失数量：", df["exp"].isna().sum())
print("无法转换为数值的数量：", exp_numeric.isna().sum())

print("\n===== 检查完成 =====")


print("\n===== 6. 无法转换为数字的 exp =====")

bad_exp = pd.to_numeric(df["exp"], errors="coerce").isna()

print(df.loc[bad_exp, ["CMPD_CHEMBLID", "exp", "smiles"]].to_string())

print("\n===== 7. 完全重复记录示例 =====")

duplicate_rows = df[df.duplicated(keep=False)]

print(duplicate_rows.head(10).to_string())

print("\n===== 8. 重复 SMILES 但记录不完全相同的示例 =====")

smiles_dup = df[
    df.duplicated(subset=["smiles"], keep=False)
].sort_values("smiles")

print(smiles_dup.head(15).to_string())


print("\n===== 9. SMILES 相同但记录内容不同的情况 =====")

# 找出所有重复 SMILES
duplicated_smiles = df[
    df["smiles"].notna()
    & df.duplicated(subset=["smiles"], keep=False)
]

# 按 SMILES 分组，找出组内其他字段存在差异的情况
different_groups = []

for smiles, group in duplicated_smiles.groupby("smiles"):
    if group[["CMPD_CHEMBLID", "exp"]].nunique(
        dropna=False
    ).gt(1).any():
        different_groups.append(group)

if different_groups:
    result = pd.concat(different_groups)
    print(result.to_string(index=False))
else:
    print("没有发现 SMILES 相同但编号或 exp 不同的记录。")

print("\n===== 10. 数据量汇总 =====")
print("原始记录数：", len(df))
print("完全重复记录数：", df.duplicated().sum())
print("重复 SMILES 的额外记录数：",
      df.duplicated(subset=["smiles"]).sum())

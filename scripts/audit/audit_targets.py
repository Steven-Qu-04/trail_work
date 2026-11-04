
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[2]

df = pd.read_csv(PROJECT_DIR / "data" / "raw" / "Lipophilicity.csv")

# 将 exp 转换为数值
df["exp_numeric"] = pd.to_numeric(
    df["exp"], errors="coerce"
)

# 排除无法转换的值，仅用于统计
valid = df.dropna(subset=["exp_numeric"])

print("===== 1. exp 基本统计 =====")
print(valid["exp_numeric"].describe())

print("\n===== 2. exp 最小的10条记录 =====")
print(
    valid.nsmallest(10, "exp_numeric")[
        ["CMPD_CHEMBLID", "exp", "smiles"]
    ].to_string(index=False)
)

print("\n===== 3. exp 最大的10条记录 =====")
print(
    valid.nlargest(10, "exp_numeric")[
        ["CMPD_CHEMBLID", "exp", "smiles"]
    ].to_string(index=False)
)

print("\n===== 4. exp 大于10的记录 =====")
print(
    valid[valid["exp_numeric"] > 10][
        ["CMPD_CHEMBLID", "exp", "smiles"]
    ].to_string(index=False)
)

print("\n===== 5. 检查完成 =====")

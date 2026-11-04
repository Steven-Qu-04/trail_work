
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[2]
INTERIM_DIR = PROJECT_DIR / "data" / "interim"
AUDIT_DIR = PROJECT_DIR / "data" / "audit"
AUDIT_DIR.mkdir(parents=True, exist_ok=True)

# 读取有效数据
df = pd.read_csv(INTERIM_DIR / "Lipophilicity_valid.csv")

# 按 SMILES 分组，保留原始标签信息
grouped = df.groupby("smiles", as_index=False).agg(
    exp=("exp", "mean"),
    exp_std=("exp", "std"),
    record_count=("exp", "size"),
    CMPD_CHEMBLID=("CMPD_CHEMBLID", lambda x: ";".join(map(str, x)))
)

# 单条记录的标准差没有定义，填0
grouped["exp_std"] = grouped["exp_std"].fillna(0)

# 保存建模数据
grouped.to_csv(INTERIM_DIR / "Lipophilicity_model.csv", index=False)

# 单独保存存在标签冲突的分子
conflicts = grouped[grouped["record_count"] > 1]
conflicts.to_csv(AUDIT_DIR / "merged_conflicts.csv", index=False)

print("===== 建模数据准备报告 =====")
print("原始有效记录数：", len(df))
print("合并后分子数量：", len(grouped))
print("合并的重复分子组数：", len(conflicts))
print("标签冲突分子数量：", (conflicts["exp_std"] > 0).sum())

print("\n冲突分子处理结果：")
print(conflicts.to_string(index=False))

print("\n已生成 Lipophilicity_model.csv")
print("已生成 merged_conflicts.csv")


from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[2]

# 读取老师提供的数据
df = pd.read_csv(PROJECT_DIR / "data" / "raw" / "Lipophilicity.csv")

# 查看数据规模
print("数据行数和列数：", df.shape)

# 查看所有列名
print("\n所有列名：")
print(df.columns.tolist())

# 查看前五行
print("\n前五行数据：")
print(df.head())

# 查看数据类型
print("\n数据类型：")
print(df.dtypes)

# 查看缺失值
print("\n每列缺失值数量：")
print(df.isnull().sum())

# 查看重复行
print("\n重复行数量：", df.duplicated().sum())

# 查看统计信息
print("\n数值列统计信息：")
print(df.describe(include="all"))

# 检查必要列
required = ["CMPD_CHEMBLID", "smiles", "exp"]

for col in required:
    if col not in df.columns:
        raise ValueError(f"缺少必要列：{col}")

print("\n数据检查完成！")

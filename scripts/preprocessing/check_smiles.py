
from pathlib import Path

import pandas as pd
from rdkit import Chem

PROJECT_DIR = Path(__file__).resolve().parents[2]
INTERIM_DIR = PROJECT_DIR / "data" / "interim"
AUDIT_DIR = PROJECT_DIR / "data" / "audit"
AUDIT_DIR.mkdir(parents=True, exist_ok=True)

# 读取清洗后的数据
df = pd.read_csv(INTERIM_DIR / "Lipophilicity_clean.csv")

valid_indices = []
invalid_records = []

for idx, row in df.iterrows():
    smiles = str(row["smiles"])
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        invalid_records.append({
            "index": idx,
            "CMPD_CHEMBLID": row["CMPD_CHEMBLID"],
            "exp": row["exp"],
            "smiles": smiles
        })
    else:
        valid_indices.append(idx)

# 只保留有效 SMILES
valid_df = df.loc[valid_indices].copy()

# 保存最终建模数据
valid_df.to_csv(INTERIM_DIR / "Lipophilicity_valid.csv", index=False)

# 保存无效记录，方便审计
pd.DataFrame(invalid_records).to_csv(AUDIT_DIR / "invalid_smiles.csv", index=False)

print("===== 最终数据检查报告 =====")
print("原始清洗后记录数：", len(df))
print("有效 SMILES 数量：", len(valid_df))
print("无效 SMILES 数量：", len(invalid_records))
print("最终数据已保存：Lipophilicity_valid.csv")
print("无效记录已保存：invalid_smiles.csv")
print("检查完成！")

"""筛除极端结构异常分子，同时保留完整审计记录。"""

from pathlib import Path

import pandas as pd
from rdkit import Chem


INPUT_PATH = Path("data/interim/Lipophilicity_model.csv")
OUTPUT_PATH = Path("data/processed/Lipophilicity_model_filtered.csv")
EXCLUDED_PATH = Path("data/audit/structural_outliers.csv")

# 当前数据中正常分子的最大重原子数为 115；两条人工超长碳链分别为
# 4612 和 6000。200 的上限在两者之间保留了充足余量。
MAX_HEAVY_ATOMS = 200


def inspect_structure(smiles: str, row_number: int) -> tuple[int, int]:
    """解析 SMILES，并返回重原子数和 SMILES 长度。"""
    if not isinstance(smiles, str) or not smiles.strip():
        raise ValueError(f"第 {row_number} 行的 SMILES 为空")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"第 {row_number} 行包含无效 SMILES：{smiles}")

    return mol.GetNumHeavyAtoms(), len(smiles)


def main() -> None:
    project_dir = Path(__file__).resolve().parents[2]
    input_path = project_dir / INPUT_PATH
    output_path = project_dir / OUTPUT_PATH
    excluded_path = project_dir / EXCLUDED_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    excluded_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"找不到输入文件：{input_path}")

    df = pd.read_csv(input_path)
    if "smiles" not in df.columns:
        raise ValueError("输入文件缺少 smiles 列")

    structure_stats = [
        inspect_structure(smiles, row_number=position + 2)
        for position, smiles in enumerate(df["smiles"].tolist())
    ]
    heavy_atom_counts = pd.Series(
        (item[0] for item in structure_stats),
        index=df.index,
        dtype="int64",
    )
    smiles_lengths = pd.Series(
        (item[1] for item in structure_stats),
        index=df.index,
        dtype="int64",
    )

    outlier_mask = heavy_atom_counts > MAX_HEAVY_ATOMS
    filtered_df = df.loc[~outlier_mask].copy()
    excluded_df = df.loc[outlier_mask].copy()
    excluded_df["heavy_atom_count"] = heavy_atom_counts.loc[outlier_mask]
    excluded_df["smiles_length"] = smiles_lengths.loc[outlier_mask]
    excluded_df["exclusion_reason"] = (
        f"heavy_atom_count > {MAX_HEAVY_ATOMS}; "
        "extreme structural outlier that causes abnormal RDKit processing"
    )

    if excluded_df.empty:
        raise RuntimeError(
            "没有检测到结构异常记录；为避免意外覆盖，请先核查输入数据和阈值"
        )

    filtered_df.to_csv(output_path, index=False)
    excluded_df.to_csv(excluded_path, index=False)

    print("===== 结构异常筛查报告 =====")
    print(f"输入记录数：{len(df)}")
    print(f"重原子数上限：{MAX_HEAVY_ATOMS}")
    print(f"排除记录数：{len(excluded_df)}")
    for _, row in excluded_df.iterrows():
        compound_id = row.get("CMPD_CHEMBLID", "未知编号")
        print(f"  {compound_id}：{int(row['heavy_atom_count'])} 个重原子")
    print(f"保留记录数：{len(filtered_df)}")
    print(f"已生成：{output_path.name}")
    print(f"已生成：{excluded_path.name}")


if __name__ == "__main__":
    main()

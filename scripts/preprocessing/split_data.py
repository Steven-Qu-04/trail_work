"""使用 Bemis-Murcko scaffold 划分分子数据集。"""

from collections import defaultdict
import hashlib
from pathlib import Path
import random

import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


RANDOM_SEED = 42
SPLIT_RATIOS = {"train": 0.8, "valid": 0.1, "scaffold_test": 0.1}
INPUT_PATH = Path("data/processed/Lipophilicity_model_filtered.csv")
OUTPUT_FILENAMES = {
    "train": Path("data/processed/train.csv"),
    "valid": Path("data/processed/valid.csv"),
    "scaffold_test": Path("data/processed/scaffold_test.csv"),
}
PROGRESS_INTERVAL = 250


def get_scaffold(smiles: str, row_number: int) -> str:
    """返回规范化的 Murcko scaffold；无环分子按规范 SMILES 分组。"""
    if not isinstance(smiles, str) or not smiles.strip():
        raise ValueError(f"第 {row_number} 行的 SMILES 为空")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"第 {row_number} 行包含无效 SMILES：{smiles}")

    # 无环分子没有 Murcko scaffold。先判断环数量可以跳过昂贵且无意义的
    # 骨架计算；哈希只用于生成紧凑、确定性的分组键，不会写入输出数据。
    if mol.GetRingInfo().NumRings() == 0:
        smiles_hash = hashlib.sha256(smiles.strip().encode("utf-8")).hexdigest()
        return "ACYCLIC_" + smiles_hash

    scaffold = MurckoScaffold.MurckoScaffoldSmiles(
        mol=mol,
        includeChirality=False,
    )

    if not scaffold:
        raise RuntimeError(f"第 {row_number} 行的含环分子未能生成 scaffold")

    return scaffold


def build_scaffold_groups(df: pd.DataFrame) -> dict[str, list[int]]:
    """提取 scaffold，并返回 scaffold 到数据行位置的映射。"""
    groups: dict[str, list[int]] = defaultdict(list)
    total = len(df)

    print(f"开始提取 {total} 个分子的 scaffold...", flush=True)
    for position, smiles in enumerate(df["smiles"].tolist()):
        # CSV 首行为表头，因此用户看到的数据行号从 2 开始。
        scaffold = get_scaffold(smiles, row_number=position + 2)
        groups[scaffold].append(position)

        completed = position + 1
        if completed % PROGRESS_INTERVAL == 0 or completed == total:
            print(
                f"  已处理 {completed}/{total} 个分子，"
                f"当前 scaffold 组数：{len(groups)}",
                flush=True,
            )

    return dict(groups)


def assign_scaffold_groups(
    scaffold_groups: dict[str, list[int]],
    total: int,
) -> tuple[dict[str, list[int]], dict[str, set[str]]]:
    """将完整 scaffold 组分配到三个集合，并尽量接近目标比例。"""
    targets = {name: ratio * total for name, ratio in SPLIT_RATIOS.items()}
    assigned_indices = {name: [] for name in SPLIT_RATIOS}
    assigned_scaffolds = {name: set() for name in SPLIT_RATIOS}
    counts = {name: 0 for name in SPLIT_RATIOS}

    # 先打乱再稳定地按组大小排序，使同样大小的组能在固定种子下随机排序。
    group_items = list(scaffold_groups.items())
    random.Random(RANDOM_SEED).shuffle(group_items)
    group_items.sort(key=lambda item: len(item[1]), reverse=True)

    for scaffold, indices in group_items:
        group_size = len(indices)
        # 选择加入当前组后相对目标完成度最低的集合，将组大小纳入决策。
        split_name = min(
            SPLIT_RATIOS,
            key=lambda name: (counts[name] + group_size) / targets[name],
        )
        assigned_indices[split_name].extend(indices)
        assigned_scaffolds[split_name].add(scaffold)
        counts[split_name] += group_size

    return assigned_indices, assigned_scaffolds


def validate_split(
    assigned_indices: dict[str, list[int]],
    assigned_scaffolds: dict[str, set[str]],
    total: int,
) -> None:
    """验证样本不重不漏，并确认三个集合之间没有 scaffold 交叉。"""
    all_indices = [
        index
        for split_indices in assigned_indices.values()
        for index in split_indices
    ]
    if len(all_indices) != total or set(all_indices) != set(range(total)):
        raise RuntimeError("划分验证失败：样本存在遗漏或重复")

    split_names = list(SPLIT_RATIOS)
    for i, left_name in enumerate(split_names):
        for right_name in split_names[i + 1 :]:
            overlap = assigned_scaffolds[left_name] & assigned_scaffolds[right_name]
            if overlap:
                raise RuntimeError(
                    f"划分验证失败：{left_name} 与 {right_name} "
                    f"存在 {len(overlap)} 个重复 scaffold"
                )


def main() -> None:
    project_dir = Path(__file__).resolve().parents[2]
    input_path = project_dir / INPUT_PATH

    if not input_path.exists():
        raise FileNotFoundError(f"找不到输入文件：{input_path}")

    print(f"读取输入文件：{input_path.name}", flush=True)
    df = pd.read_csv(input_path)

    required_columns = {"smiles", "exp"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"输入文件缺少必要列：{sorted(missing_columns)}")
    if df.empty:
        raise ValueError("输入文件没有数据")
    if df["smiles"].duplicated().any():
        duplicate_count = int(df["smiles"].duplicated().sum())
        raise ValueError(
            f"输入文件仍有 {duplicate_count} 条重复 SMILES，请先核查建模数据"
        )

    scaffold_groups = build_scaffold_groups(df)
    assigned_indices, assigned_scaffolds = assign_scaffold_groups(
        scaffold_groups,
        total=len(df),
    )
    validate_split(assigned_indices, assigned_scaffolds, total=len(df))

    split_frames = {
        name: df.iloc[indices].copy().reset_index(drop=True)
        for name, indices in assigned_indices.items()
    }

    print("\n划分验证通过：样本无遗漏、无重复，集合间无 scaffold 交叉。")
    print("===== Scaffold Split 报告 =====")
    print(f"总分子数：{len(df)}")
    print(f"scaffold 组数：{len(scaffold_groups)}")

    display_names = {
        "train": "训练集",
        "valid": "验证集",
        "scaffold_test": "内部 Scaffold 测试集",
    }
    for name, frame in split_frames.items():
        print(
            f"{display_names[name]}：{len(frame)} 条 "
            f"({len(frame) / len(df):.2%})，"
            f"scaffold 组数：{len(assigned_scaffolds[name])}"
        )

    # 所有验证通过后再写文件，避免失败时产生不完整的划分结果。
    for name, frame in split_frames.items():
        output_path = project_dir / OUTPUT_FILENAMES[name]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output_path, index=False)
        print(f"已生成：{output_path.name}")

    print("Scaffold Split 完成！")


if __name__ == "__main__":
    main()

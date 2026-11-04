"""训练和预测共用的 Morgan 指纹特征提取函数。"""

from functools import lru_cache
from typing import Iterable

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, rdFingerprintGenerator, rdMolDescriptors


DEFAULT_RADIUS = 2
DEFAULT_FP_SIZE = 2048
DEFAULT_DESCRIPTOR_NAMES = (
    "MolWt",
    "TPSA",
    "NumHDonors",
    "NumHAcceptors",
    "NumRotatableBonds",
    "RingCount",
    "NumAromaticRings",
    "FractionCSP3",
    "HeavyAtomCount",
    "FormalCharge",
)

DESCRIPTOR_FUNCTIONS = {
    "MolWt": Descriptors.MolWt,
    "TPSA": rdMolDescriptors.CalcTPSA,
    "NumHDonors": rdMolDescriptors.CalcNumHBD,
    "NumHAcceptors": rdMolDescriptors.CalcNumHBA,
    "NumRotatableBonds": rdMolDescriptors.CalcNumRotatableBonds,
    "RingCount": rdMolDescriptors.CalcNumRings,
    "NumAromaticRings": rdMolDescriptors.CalcNumAromaticRings,
    "FractionCSP3": rdMolDescriptors.CalcFractionCSP3,
    "HeavyAtomCount": lambda mol: mol.GetNumHeavyAtoms(),
    "FormalCharge": Chem.GetFormalCharge,
}


@lru_cache(maxsize=None)
def get_morgan_generator(radius: int, fp_size: int):
    """按配置创建并缓存 Morgan 指纹生成器。"""
    if radius < 0:
        raise ValueError("radius 不能小于 0")
    if fp_size <= 0:
        raise ValueError("fp_size 必须大于 0")
    return rdFingerprintGenerator.GetMorganGenerator(
        radius=radius,
        fpSize=fp_size,
    )


def smiles_to_mol(smiles: str):
    """将有效的非空 SMILES 转换为 RDKit Mol；无效时返回 None。"""
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    return Chem.MolFromSmiles(smiles)


def mol_to_fingerprint(
    mol,
    radius: int = DEFAULT_RADIUS,
    fp_size: int = DEFAULT_FP_SIZE,
):
    """将 Mol 转换为固定长度的 uint8 Morgan 位指纹。"""
    if mol is None:
        return None

    generator = get_morgan_generator(radius, fp_size)
    bit_vector = generator.GetFingerprint(mol)
    fingerprint = np.zeros(fp_size, dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(bit_vector, fingerprint)
    return fingerprint


def mol_to_count_fingerprint(
    mol,
    radius: int = DEFAULT_RADIUS,
    fp_size: int = DEFAULT_FP_SIZE,
):
    """将 Mol 转换为固定长度的 Morgan 计数指纹。"""
    if mol is None:
        return None

    generator = get_morgan_generator(radius, fp_size)
    count_vector = generator.GetCountFingerprint(mol)
    fingerprint = np.zeros(fp_size, dtype=np.int32)
    DataStructs.ConvertToNumpyArray(count_vector, fingerprint)
    return fingerprint


def smiles_to_fingerprint(
    smiles: str,
    radius: int = DEFAULT_RADIUS,
    fp_size: int = DEFAULT_FP_SIZE,
):
    """使用指定配置将 SMILES 转换为 Morgan 指纹。"""
    return mol_to_fingerprint(
        smiles_to_mol(smiles),
        radius=radius,
        fp_size=fp_size,
    )


def smiles_list_to_matrix(
    smiles_values: Iterable[str],
    radius: int = DEFAULT_RADIUS,
    fp_size: int = DEFAULT_FP_SIZE,
) -> tuple[np.ndarray, list[int]]:
    """批量生成特征矩阵，并返回无法解析的输入位置。"""
    fingerprints = []
    invalid_positions = []

    for position, smiles in enumerate(smiles_values):
        fingerprint = smiles_to_fingerprint(
            smiles,
            radius=radius,
            fp_size=fp_size,
        )
        if fingerprint is None:
            invalid_positions.append(position)
        else:
            fingerprints.append(fingerprint)

    if fingerprints:
        matrix = np.stack(fingerprints)
    else:
        matrix = np.empty((0, fp_size), dtype=np.uint8)

    return matrix, invalid_positions


def mol_to_descriptors(
    mol,
    descriptor_names: Iterable[str] = DEFAULT_DESCRIPTOR_NAMES,
):
    """按固定顺序计算不包含 MolLogP 的基础理化描述符。"""
    if mol is None:
        return None

    names = tuple(descriptor_names)
    unknown_names = [name for name in names if name not in DESCRIPTOR_FUNCTIONS]
    if unknown_names:
        raise ValueError(f"不支持的描述符：{unknown_names}")

    return np.asarray(
        [DESCRIPTOR_FUNCTIONS[name](mol) for name in names],
        dtype=np.float32,
    )


def smiles_list_to_feature_matrix(
    smiles_values: Iterable[str],
    feature_config: dict,
) -> tuple[np.ndarray, list[int]]:
    """根据模型保存的配置生成 Morgan 或 Morgan+描述符特征矩阵。"""
    feature_type = feature_config.get("type", "Morgan")
    radius = int(feature_config.get("radius", DEFAULT_RADIUS))
    fp_size = int(feature_config.get("fp_size", DEFAULT_FP_SIZE))

    if feature_type == "Morgan":
        return smiles_list_to_matrix(
            smiles_values,
            radius=radius,
            fp_size=fp_size,
        )
    supported_types = {
        "RDKitDescriptors",
        "MorganCount",
        "Morgan+RDKitDescriptors",
        "MorganCount+RDKitDescriptors",
    }
    if feature_type not in supported_types:
        raise ValueError(f"不支持的特征类型：{feature_type}")

    descriptor_names = tuple(
        feature_config.get("descriptor_names", DEFAULT_DESCRIPTOR_NAMES)
    )
    rows = []
    invalid_positions = []

    for position, smiles in enumerate(smiles_values):
        mol = smiles_to_mol(smiles)
        if mol is None:
            invalid_positions.append(position)
            continue

        descriptors = mol_to_descriptors(mol, descriptor_names)
        if feature_type == "RDKitDescriptors":
            rows.append(descriptors)
            continue

        if feature_type.startswith("MorganCount"):
            fingerprint = mol_to_count_fingerprint(
                mol,
                radius=radius,
                fp_size=fp_size,
            )
        else:
            fingerprint = mol_to_fingerprint(
                mol,
                radius=radius,
                fp_size=fp_size,
            )

        if feature_type in {"MorganCount", "Morgan"}:
            rows.append(fingerprint.astype(np.float32))
        else:
            rows.append(
                np.concatenate(
                    [fingerprint.astype(np.float32), descriptors],
                    axis=0,
                )
            )

    if feature_type == "RDKitDescriptors":
        width = len(descriptor_names)
    elif feature_type in {"MorganCount", "Morgan"}:
        width = fp_size
    else:
        width = fp_size + len(descriptor_names)
    if rows:
        matrix = np.stack(rows).astype(np.float32, copy=False)
    else:
        matrix = np.empty((0, width), dtype=np.float32)

    return matrix, invalid_positions

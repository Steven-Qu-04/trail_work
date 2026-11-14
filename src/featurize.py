"""featurize.py — 分子特征的唯一实现

03 训练前的特征、07 预测新分子，都必须走这里的函数，
否则两份代码会出现漂移（描述符顺序不一致、指纹参数不同），预测结果直接错。

对外接口：
  mols_from_smiles(list) -> list[Mol 或 None]
  morgan_matrix(mols)     -> (n, 2048) uint8，无法解析的分子给全 0
  desc_frame(mols)        -> (n, 12) DataFrame，无法解析的分子给 NaN
  desc_names()            -> 12 个描述符的名字（顺序与 desc_frame 一致）

注意：SMILES 超过 MAX_SMILES_LEN 要先拦掉再交给 RDKit，
几千个原子的长链会让 RDKit 递归爆栈、整个 Python 进程崩掉（0xC00000FD）。
"""

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Crippen, Descriptors, Lipinski, rdMolDescriptors
from rdkit.DataStructs import ConvertToNumpyArray

RDLogger.DisableLog("rdApp.*")

MORGAN_RADIUS = 2
MORGAN_BITS = 2048
MAX_SMILES_LEN = 300

DESCRIPTORS = [
    ("MolWt", lambda m: Descriptors.MolWt(m)),
    ("MolLogP", lambda m: Crippen.MolLogP(m)),
    ("TPSA", lambda m: rdMolDescriptors.CalcTPSA(m)),
    ("NumHDonors", lambda m: Lipinski.NumHDonors(m)),
    ("NumHAcceptors", lambda m: Lipinski.NumHAcceptors(m)),
    ("NumRotatableBonds", lambda m: rdMolDescriptors.CalcNumRotatableBonds(m)),
    ("RingCount", lambda m: rdMolDescriptors.CalcNumRings(m)),
    ("NumAromaticRings", lambda m: rdMolDescriptors.CalcNumAromaticRings(m)),
    ("FractionCSP3", lambda m: rdMolDescriptors.CalcFractionCSP3(m)),
    ("HeavyAtomCount", lambda m: m.GetNumHeavyAtoms()),
    ("NumHeteroatoms", lambda m: rdMolDescriptors.CalcNumHeteroatoms(m)),
    ("BertzCT", lambda m: Descriptors.BertzCT(m)),
]


def desc_names():
    return [name for name, _ in DESCRIPTORS]


def mols_from_smiles(smiles_list):
    """把 SMILES 转成 RDKit 分子；空串、超长、解析失败的一律返回 None。"""
    mols = []
    for smi in smiles_list:
        if not isinstance(smi, str) or not smi.strip() or len(smi) > MAX_SMILES_LEN:
            mols.append(None)
            continue
        mols.append(Chem.MolFromSmiles(smi))
    return mols


def morgan_matrix(mols, radius=MORGAN_RADIUS, n_bits=MORGAN_BITS):
    """Morgan 指纹（ECFP4）。解析失败的分子给全 0 行，调用方自己决定是否剔除。"""
    X = np.zeros((len(mols), n_bits), dtype=np.uint8)
    buf = np.zeros((n_bits,), dtype=np.int8)
    for i, mol in enumerate(mols):
        if mol is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
        ConvertToNumpyArray(fp, buf)
        X[i] = buf
    return X


def desc_frame(mols):
    """12 个描述符。解析失败的分子整行 NaN，由模型侧的填补器处理。"""
    table = pd.DataFrame(index=range(len(mols)))
    for name, fn in DESCRIPTORS:
        vals = []
        for mol in mols:
            if mol is None:
                vals.append(np.nan)
                continue
            try:
                vals.append(float(fn(mol)))
            except Exception:
                vals.append(np.nan)
        table[name] = vals
    return table.replace([np.inf, -np.inf], np.nan)

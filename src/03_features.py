"""03_features.py — 分子特征生成

把每个分子变成模型能吃的数字。两类特征：

  1. Morgan 指纹（ECFP4）：Morgan 半径 2、2048 位，每个分子一个 0/1 向量
     —— 记录"有哪些局部子结构"，是分子性质预测最常用的特征
  2. 分子描述符：分子量、logP、氢键供受体、TPSA、可旋转键、环数等 12 个
     —— 记录"整体物理化学性质"，对 logD 这种亲疏水性标签很关键

输出（都在 data/ 下，行序与 data/clean.csv 完全一致）：
  X_morgan.npy     (3780, 2048) uint8
  X_desc.npy       (3780, 12)   float32
  y.npy            (3780,)      float32
  desc_cols.json   描述符列名与生成参数
  meta.csv         编号 + canonical SMILES，用于对齐

运行：python src/03_features.py
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Crippen, Descriptors, Lipinski, rdMolDescriptors
from rdkit.DataStructs import ConvertToNumpyArray

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[1]
CLEAN_CSV = ROOT / "data" / "clean.csv"
OUT_DIR = ROOT / "data"
LOG_MD = ROOT / "reports" / "features_log.md"

MORGAN_RADIUS = 2
MORGAN_BITS = 2048

# 12 个描述符：选的都是对脂溶性有直接解释力的，别一股脑塞几百个
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


def main() -> None:
    if not CLEAN_CSV.exists():
        raise SystemExit("找不到 data/clean.csv，先运行：python src/02_clean.py")

    df = pd.read_csv(CLEAN_CSV)
    n = len(df)

    mols = [Chem.MolFromSmiles(s) for s in df["canon_smiles"]]
    bad = [i for i, m in enumerate(mols) if m is None]
    if bad:
        raise SystemExit(f"clean.csv 里有 {len(bad)} 行无法解析，清洗步骤有问题")

    # ---- Morgan 指纹 ----
    X_morgan = np.zeros((n, MORGAN_BITS), dtype=np.uint8)
    buf = np.zeros((MORGAN_BITS,), dtype=np.int8)
    for i, mol in enumerate(mols):
        fp = AllChem.GetMorganFingerprintAsBitVect(
            mol, radius=MORGAN_RADIUS, nBits=MORGAN_BITS
        )
        ConvertToNumpyArray(fp, buf)
        X_morgan[i] = buf

    # ---- 描述符 ----
    X_desc = pd.DataFrame(index=range(n))
    for name, fn in DESCRIPTORS:
        vals = []
        for mol in mols:
            try:
                vals.append(float(fn(mol)))
            except Exception:
                vals.append(np.nan)
        X_desc[name] = vals
    X_desc = X_desc.replace([np.inf, -np.inf], np.nan)

    y = df["exp"].to_numpy(dtype=np.float32)

    # ---- 保存 ----
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUT_DIR / "X_morgan.npy", X_morgan)
    np.save(OUT_DIR / "X_desc.npy", X_desc.to_numpy(dtype=np.float32))
    np.save(OUT_DIR / "y.npy", y)
    (OUT_DIR / "desc_cols.json").write_text(
        json.dumps(
            {
                "descriptors": [name for name, _ in DESCRIPTORS],
                "morgan_radius": MORGAN_RADIUS,
                "morgan_bits": MORGAN_BITS,
                "n_samples": n,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    df[["CMPD_CHEMBLID", "canon_smiles"]].to_csv(
        OUT_DIR / "meta.csv", index=False, encoding="utf-8"
    )

    # ---- 自检 ----
    row_sums = X_morgan.sum(axis=1)
    checks = [
        ("指纹形状正确", X_morgan.shape == (n, MORGAN_BITS)),
        ("每个分子至少有一个指纹位", bool((row_sums > 0).all())),
        ("描述符形状正确", X_desc.shape == (n, len(DESCRIPTORS))),
        ("标签行数与 clean.csv 一致", len(y) == n),
        ("标签都在合理范围", bool(np.abs(y).max() <= 10.0)),
    ]
    ok = all(v for _, v in checks)

    # ---- 报告 ----
    md = []
    md.append("# 特征生成报告")
    md.append("")
    md.append(f"- 样本数：**{n}**")
    md.append(f"- 指纹：Morgan（ECFP4），半径 {MORGAN_RADIUS}，{MORGAN_BITS} 位")
    md.append(f"- 描述符：{len(DESCRIPTORS)} 个")
    md.append("")
    md.append("## 1. 描述符统计")
    md.append("")
    md.append("| 描述符 | 缺失数 | 最小值 | 中位数 | 最大值 |")
    md.append("| --- | --- | --- | --- | --- |")
    for name in X_desc.columns:
        col = X_desc[name]
        md.append(
            f"| `{name}` | {int(col.isna().sum())} | {col.min():.3f} | "
            f"{col.median():.3f} | {col.max():.3f} |"
        )
    md.append("")
    md.append("> 描述符里的缺失值（NaN）不在这一步填，留给模型 pipeline 用训练集统计量填补，"
              "避免验证集/测试集的信息泄漏进来。")
    md.append("")
    nan_desc = int(X_desc.isna().sum().sum())
    md.append(f"描述符空值合计：**{nan_desc}** 个单元格"
              + ("（下一步会用训练集的中位数填补）" if nan_desc else "（没有缺失）"))
    md.append("")
    md.append("## 2. 指纹稀疏度")
    md.append("")
    md.append(f"- 每个分子平均激活 {X_morgan.sum(axis=1).mean():.1f} 个位"
              f"（共 {MORGAN_BITS} 位）")
    md.append(f"- 激活最少的分子：{int(row_sums.min())} 位；最多：{int(row_sums.max())} 位")
    md.append("")
    md.append("## 3. 输出文件")
    md.append("")
    md.append("| 文件 | 形状 | 说明 |")
    md.append("| --- | --- | --- |")
    md.append(f"| `data/X_morgan.npy` | {X_morgan.shape} | Morgan 指纹，uint8 |")
    md.append(f"| `data/X_desc.npy` | {X_desc.shape} | 12 个描述符，float32 |")
    md.append(f"| `data/y.npy` | {y.shape} | logD 标签 |")
    md.append("| `data/desc_cols.json` | — | 描述符列名与参数，保证复现 |")
    md.append("| `data/meta.csv` | — | 编号 + SMILES，行序与上面三个数组一致 |")
    md.append("")
    md.append("## 4. 自检结果")
    md.append("")
    for name, passed in checks:
        md.append(f"- {'通过' if passed else '未通过'}：{name}")
    md.append("")
    md.append(f"总体：**{'通过' if ok else '未通过'}**")
    md.append("")

    LOG_MD.write_text("\n".join(md), encoding="utf-8")

    print(f"样本数    ：{n}")
    print(f"指纹      ：{X_morgan.shape}，平均激活 {row_sums.mean():.1f} 位")
    print(f"描述符    ：{X_desc.shape}，空值 {nan_desc} 个")
    print(f"标签      ：{y.shape}，{y.min():.2f} ~ {y.max():.2f}")
    print(f"报告      ：{LOG_MD}")
    for name, passed in checks:
        print(f"自检 {name}：{'通过' if passed else '未通过'}")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

"""04_split.py — 按分子骨架划分训练/验证/测试集

为什么不能随机划分：同一骨架的分子（比如同一个母核上换取代基）性质高度相似，
随机划分会让它们同时出现在训练集和测试集，模型只要"记住"这个骨架就能拿高分，
测出来的 RMSE 是虚低的。按骨架（Murcko scaffold）分组划分，测试集里的骨架
在训练时完全没见过，才叫真正的泛化。

规则：
  - 骨架 = Murcko scaffold（去掉侧链、只留环系和连接骨架）
  - 无环分子没有骨架，每个分子单独成组（否则所有链状分子会被塞进同一组）
  - 按骨架大小从大到小分组，依次填训练集到 80%、验证集到 10%、剩下给测试集
  - 固定随机种子，任何人重跑结果完全一致

输出：
  data/split.npz    train_idx / val_idx / test_idx（整数下标，对应 clean.csv 行序）
  data/split.csv    CMPD_CHEMBLID, scaffold, split
  reports/split_log.md

运行：python src/04_split.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[1]
CLEAN_CSV = ROOT / "data" / "clean.csv"
OUT_NPZ = ROOT / "data" / "split.npz"
OUT_CSV = ROOT / "data" / "split.csv"
LOG_MD = ROOT / "reports" / "split_log.md"

TRAIN_FRAC = 0.8
VAL_FRAC = 0.1
SEED = 42


def main() -> None:
    if not CLEAN_CSV.exists():
        raise SystemExit("找不到 data/clean.csv，先运行：python src/02_clean.py")

    df = pd.read_csv(CLEAN_CSV)
    n = len(df)

    scaffolds, n_acyclic = [], 0
    for smi in df["canon_smiles"]:
        scaf = MurckoScaffold.MurckoScaffoldSmiles(smiles=smi)
        if not scaf:
            scaf = "ACYCLIC:" + smi
            n_acyclic += 1
        scaffolds.append(scaf)
    df["scaffold"] = scaffolds

    # ---- 按骨架分组，大组优先 ----
    groups = df.groupby("scaffold").indices
    order = sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0]))

    train_idx, val_idx, test_idx = [], [], []
    for _, idx in order:
        if len(train_idx) < TRAIN_FRAC * n:
            train_idx.extend(idx.tolist())
        elif len(val_idx) < VAL_FRAC * n:
            val_idx.extend(idx.tolist())
        else:
            test_idx.extend(idx.tolist())

    train_idx = np.array(sorted(train_idx))
    val_idx = np.array(sorted(val_idx))
    test_idx = np.array(sorted(test_idx))

    # ---- 对照：随机划分会漏成什么样 ----
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(n)
    n_tr = int(TRAIN_FRAC * n)
    n_va = int(VAL_FRAC * n)
    r_train = perm[:n_tr]
    r_test = perm[n_tr + n_va:]
    rand_overlap = len(set(df["scaffold"].iloc[r_train]) & set(df["scaffold"].iloc[r_test]))

    # ---- 自检 ----
    all_idx = np.concatenate([train_idx, val_idx, test_idx])
    split_sets = {
        "train": set(df["scaffold"].iloc[train_idx]),
        "val": set(df["scaffold"].iloc[val_idx]),
        "test": set(df["scaffold"].iloc[test_idx]),
    }
    checks = [
        ("三个集合覆盖全部样本", len(all_idx) == n and len(set(all_idx.tolist())) == n),
        ("三个集合互不重叠",
         len(set(train_idx.tolist()) & set(val_idx.tolist())) == 0
         and len(set(train_idx.tolist()) & set(test_idx.tolist())) == 0
         and len(set(val_idx.tolist()) & set(test_idx.tolist())) == 0),
        ("训练集骨架与测试集不重叠", len(split_sets["train"] & split_sets["test"]) == 0),
        ("训练集骨架与验证集不重叠", len(split_sets["train"] & split_sets["val"]) == 0),
        ("测试集骨架与验证集不重叠", len(split_sets["test"] & split_sets["val"]) == 0),
        ("三个集合都非空",
         len(train_idx) > 0 and len(val_idx) > 0 and len(test_idx) > 0),
    ]
    ok = all(v for _, v in checks)

    # ---- 保存 ----
    OUT_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT_NPZ, train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)
    df["split"] = "train"
    df.loc[val_idx, "split"] = "val"
    df.loc[test_idx, "split"] = "test"
    df[["CMPD_CHEMBLID", "scaffold", "split"]].to_csv(
        OUT_CSV, index=False, encoding="utf-8"
    )

    # ---- 报告 ----
    md = []
    md.append("# 数据集划分报告")
    md.append("")
    md.append(f"- 划分方式：按 Murcko scaffold 分组划分")
    md.append(f"- 目标比例：训练 {TRAIN_FRAC:.0%} / 验证 {VAL_FRAC:.0%} / 测试 {1 - TRAIN_FRAC - VAL_FRAC:.0%}")
    md.append(f"- 随机种子：{SEED}（骨架大小相同者按骨架字符串排序，结果可完全复现）")
    md.append(f"- 骨架总数：**{len(groups)}**（无环分子 {n_acyclic} 个，各自单独成组）")
    md.append("")
    md.append("## 1. 划分结果")
    md.append("")
    md.append("| 集合 | 样本数 | 占比 | 骨架数 | 标签均值 | 标签标准差 |")
    md.append("| --- | --- | --- | --- | --- | --- |")
    for name, idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        sub = df.iloc[idx]
        md.append(f"| {name} | {len(idx)} | {len(idx) / n:.1%} | {sub['scaffold'].nunique()} | "
                  f"{sub['exp'].mean():.2f} | {sub['exp'].std():.2f} |")
    md.append("")
    md.append(f"合计 {len(train_idx) + len(val_idx) + len(test_idx)} 条。"
              "占比不会精确等于 80/10/10——骨架是整组分配，不能拆开。")
    md.append("")
    md.append("## 2. 和随机划分的对比")
    md.append("")
    md.append(f"同样按 8:1:1 随机划分一次，训练集和测试集**共同拥有的骨架有 "
              f"{rand_overlap} 个**。按骨架划分是 **0** 个。")
    md.append("")
    md.append("> 这就是作业要求 scaffold split 的原因：随机划分下测试集里全是见过的骨架，"
              "RMSE 会明显偏低，报上去等于虚报性能。")
    md.append("")
    md.append("## 3. 输出文件")
    md.append("")
    md.append("| 文件 | 说明 |")
    md.append("| --- | --- |")
    md.append("| `data/split.npz` | `train_idx` / `val_idx` / `test_idx`，下标对应 clean.csv 行序 |")
    md.append("| `data/split.csv` | 每个分子的骨架与所在集合，便于检查 |")
    md.append("")
    md.append("## 4. 自检结果")
    md.append("")
    for name, passed in checks:
        md.append(f"- {'通过' if passed else '未通过'}：{name}")
    md.append("")
    md.append(f"总体：**{'通过' if ok else '未通过'}**")
    md.append("")

    LOG_MD.write_text("\n".join(md), encoding="utf-8")

    print(f"骨架总数  ：{len(groups)}（无环 {n_acyclic} 个单独成组）")
    print(f"训练集    ：{len(train_idx)} 行（{len(train_idx) / n:.1%}），骨架 {len(split_sets['train'])} 个")
    print(f"验证集    ：{len(val_idx)} 行（{len(val_idx) / n:.1%}），骨架 {len(split_sets['val'])} 个")
    print(f"测试集    ：{len(test_idx)} 行（{len(test_idx) / n:.1%}），骨架 {len(split_sets['test'])} 个")
    print(f"随机划分对照：训练与测试共用骨架 {rand_overlap} 个（骨架划分为 0）")
    print(f"报告      ：{LOG_MD}")
    for name, passed in checks:
        print(f"自检 {name}：{'通过' if passed else '未通过'}")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

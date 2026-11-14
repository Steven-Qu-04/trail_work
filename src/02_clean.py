"""02_clean.py — 数据清洗流水线

把 01_check_data.py 里查出来的问题，变成一套可重复执行的清洗规则：
  1. 标签必须能转成数值，且 |exp| <= 10
  2. SMILES 不能为空、不能超过 300 字符（超长会解析爆栈）
  3. SMILES 必须能被 RDKit 解析
  4. 按 canonical SMILES 去重，保留最先出现的一条

输出：
  data/clean.csv        干净数据（CMPD_CHEMBLID, exp, smiles, canon_smiles）
  reports/clean_log.md  清洗日志（剔除了什么、去掉了多少、还有哪些坑）

运行：python src/02_clean.py
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC_CSV = ROOT / "Lipophilicity.csv"
OUT_CSV = ROOT / "data" / "clean.csv"
LOG_MD = ROOT / "reports" / "clean_log.md"

MAX_SMILES_LEN = 300
EXP_ABS_LIMIT = 10.0

try:
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False


def num(x, nd=3):
    return "—" if x is None or pd.isna(x) else f"{x:.{nd}f}"


def main() -> None:
    if not HAS_RDKIT:
        raise SystemExit("缺少 rdkit，请先在 trail 环境里安装：conda install -c conda-forge rdkit")

    raw = pd.read_csv(SRC_CSV, dtype=str, keep_default_na=False)
    n0 = len(raw)

    # ---- 逐项打标 ----
    raw["exp_num"] = pd.to_numeric(raw["exp"], errors="coerce")

    empty_exp = raw["exp_num"].isna()
    extreme = raw["exp_num"].abs() > EXP_ABS_LIMIT
    empty_smiles = raw["smiles"].str.strip() == ""
    too_long = raw["smiles"].str.len() > MAX_SMILES_LEN

    # 超长 SMILES 在解析前就跳过：RDKit 处理几千原子的链会递归爆栈，
    # 进程直接 0xC00000FD 崩溃，异常都抓不到。
    canon = []
    for s, skip in zip(raw["smiles"], too_long):
        if not s.strip() or skip:
            canon.append(None)
            continue
        mol = Chem.MolFromSmiles(s)
        canon.append(Chem.MolToSmiles(mol) if mol is not None else None)
    raw["canon_smiles"] = canon

    parse_fail = raw["canon_smiles"].isna() & (~empty_smiles) & (~too_long)

    bad = empty_exp | extreme | empty_smiles | too_long | parse_fail
    dropped = raw[bad]
    kept = raw[~bad].copy()
    n_after_drop = len(kept)

    # ---- 按结构去重 ----
    n_conflict_groups = 0
    conflicts = []
    dup_groups = kept["canon_smiles"].duplicated()
    if dup_groups.any():
        grp = kept.groupby("canon_smiles")["exp_num"].nunique()
        conflict_ids = grp[grp > 1].index.tolist()
        n_conflict_groups = len(conflict_ids)
        for sid in conflict_ids:
            rows = kept[kept["canon_smiles"] == sid]
            conflicts.append((rows["CMPD_CHEMBLID"].iloc[0],
                              [num(v, 2) for v in rows["exp_num"]],
                              len(rows)))

    before = len(kept)
    clean = kept.drop_duplicates(subset="canon_smiles", keep="first")
    n_dedup = before - len(clean)

    out = clean[["CMPD_CHEMBLID", "exp_num", "smiles", "canon_smiles"]].rename(
        columns={"exp_num": "exp"}
    )
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False, encoding="utf-8")

    # ---- 自检：读回来重新验一遍 ----
    chk = pd.read_csv(OUT_CSV)
    checks = [
        ("行数与内存一致", len(chk) == len(clean)),
        ("标签无缺失", int(chk["exp"].isna().sum()) == 0),
        ("canon_smiles 唯一", bool(chk["canon_smiles"].is_unique)),
        ("标签都在合理范围", bool(chk["exp"].abs().max() <= EXP_ABS_LIMIT)),
    ]
    ok = all(v for _, v in checks)

    # ---- 写日志 ----
    md = []
    md.append("# 清洗日志")
    md.append("")
    md.append("| 项目 | 数值 |")
    md.append("| --- | --- |")
    md.append(f"| 输入行数 | {n0} |")
    md.append(f"| 剔除异常行 | {len(dropped)} |")
    md.append(f"| 去重丢弃 | {n_dedup} |")
    md.append(f"| 输出行数 | {len(out)} |")
    md.append("")
    md.append("## 1. 剔除明细（按原因计数）")
    md.append("")
    md.append("| 原因 | 行数 |")
    md.append("| --- | --- |")
    md.append(f"| 标签为空或非数值 | {int(empty_exp.sum())} |")
    md.append(f"| 标签绝对值超过 {EXP_ABS_LIMIT:g} | {int(extreme.sum())} |")
    md.append(f"| SMILES 为空 | {int(empty_smiles.sum())} |")
    md.append(f"| SMILES 超过 {MAX_SMILES_LEN} 字符 | {int(too_long.sum())} |")
    md.append(f"| SMILES 无法被 RDKit 解析 | {int(parse_fail.sum())} |")
    md.append("")
    md.append("被剔除的行：")
    md.append("")
    md.append("| 行号 | CMPD_CHEMBLID | exp 原值 | 原因 |")
    md.append("| --- | --- | --- | --- |")
    for i in dropped.index:
        why = []
        if empty_exp[i]:
            why.append("标签为空或非数值")
        if extreme[i]:
            why.append("标签越界")
        if empty_smiles[i]:
            why.append("SMILES 为空")
        if too_long[i]:
            why.append(f"SMILES 过长（{len(dropped.at[i, 'smiles'])} 字符）")
        if parse_fail[i]:
            why.append("SMILES 无法解析")
        e = dropped.at[i, "exp"].strip()
        md.append(f"| {i + 2} | `{dropped.at[i, 'CMPD_CHEMBLID']}` | "
                  f"`{e if e else '（空）'}` | {'；'.join(why)} |")
    md.append("")
    md.append("## 2. 去重")
    md.append("")
    md.append(f"- 去重前 {before} 行，去重后 {len(clean)} 行，丢弃 {n_dedup} 行")
    md.append("- 判重依据：`canon_smiles`（RDKit 规范化后的 SMILES），保留原始顺序里的第一条")
    md.append(f"- 同一结构标签矛盾的结构：**{n_conflict_groups}** 个（去重时保留首条，其余丢弃）")
    md.append("")
    if conflicts:
        md.append("| 保留的 ID | 该结构的全部标签 | 行数 |")
        md.append("| --- | --- | --- |")
        for cid, vals, cnt in conflicts:
            md.append(f"| `{cid}` | {' / '.join(vals)} | {cnt} |")
        md.append("")
        md.append("> 矛盾标签无法判断谁对，统一保留第一条，并在 README 里如实说明。")
        md.append("")
    md.append("## 3. 输出文件说明")
    md.append("")
    md.append(f"`data/clean.csv`，共 {len(out)} 行 4 列：")
    md.append("")
    md.append("| 列名 | 含义 |")
    md.append("| --- | --- |")
    md.append("| `CMPD_CHEMBLID` | 原始 ChEMBL 编号，保留像 `ljiohohoihoa` 这种乱码 ID，它只是编号不规范 |")
    md.append("| `exp` | logD 标签，已转成 float |")
    md.append("| `smiles` | 原始 SMILES，保留以便溯源 |")
    md.append("| `canon_smiles` | RDKit 规范化后的 SMILES，**建模、划分、去重都应以它为准** |")
    md.append("")
    md.append("## 4. 清洗后的标签分布")
    md.append("")
    md.append("| 指标 | 值 |")
    md.append("| --- | --- |")
    md.append(f"| 样本数 | {len(clean)} |")
    md.append(f"| 最小值 | {num(clean['exp_num'].min())} |")
    md.append(f"| 中位数 | {num(clean['exp_num'].median())} |")
    md.append(f"| 最大值 | {num(clean['exp_num'].max())} |")
    md.append(f"| 均值 | {num(clean['exp_num'].mean())} |")
    md.append(f"| 标准差 | {num(clean['exp_num'].std())} |")
    md.append("")
    md.append("## 5. 自检结果")
    md.append("")
    for name, passed in checks:
        md.append(f"- {'通过' if passed else '未通过'}：{name}")
    md.append("")
    md.append(f"总体：**{'通过' if ok else '未通过'}**")
    md.append("")

    LOG_MD.write_text("\n".join(md), encoding="utf-8")

    # ---- 控制台 ----
    print(f"输入      ：{n0} 行")
    print(f"剔除异常  ：{len(dropped)} 行")
    print(f"去重丢弃  ：{n_dedup} 行（矛盾标签结构 {n_conflict_groups} 个）")
    print(f"输出      ：{len(out)} 行 → {OUT_CSV}")
    print(f"日志      ：{LOG_MD}")
    for name, passed in checks:
        print(f"自检 {name}：{'通过' if passed else '未通过'}")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

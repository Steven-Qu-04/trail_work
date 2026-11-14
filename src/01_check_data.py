"""01_check_data.py — Lipophilicity 数据集体检

建模之前先把三列（ID / 标签 exp / SMILES）都检查一遍：
缺失、类型、取值范围、重复、无效分子、长度异常。

输出：reports/data_check.md（可直接对照写 README 第 2 节）

运行：python src/01_check_data.py
"""

from pathlib import Path
import re

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "Lipophilicity.csv"
REPORT_PATH = ROOT / "reports" / "data_check.md"

ID_PATTERN = re.compile(r"^CHEMBL\d+$")
MAX_SMILES_LEN = 300      # 正常分子最长约 270，超过即视为异常
EXP_ABS_LIMIT = 10.0      # logD 的合理量级，超出即视为异常值

try:
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    HAS_RDKIT = True
except ImportError:       # 没装 rdkit 时退化成纯文本检查
    Chem = None
    HAS_RDKIT = False


def num(x, nd=3):
    """把数值格式化成固定小数，None 显示为 —"""
    return "—" if x is None or pd.isna(x) else f"{x:.{nd}f}"


def main() -> None:
    raw = pd.read_csv(CSV_PATH, dtype=str, keep_default_na=False)
    n = len(raw)

    md = []
    md.append("# 数据体检报告")
    md.append("")
    md.append("- 数据文件：`Lipophilicity.csv`")
    md.append(f"- 行数 **{n}**，列数 **{raw.shape[1]}**")
    md.append("- 列名：" + "、".join(f"`{c}`" for c in raw.columns))
    md.append("")
    md.append("> 读入时全部按字符串处理：`exp` 列里存在 `.05` 这种无前导零的写法，"
              "直接让 pandas 推断类型会把整列变成 object。")
    md.append("")

    # ---------- 1. 缺失 ----------
    md.append("## 1. 缺失值")
    md.append("")
    md.append("| 列 | 空值行数 |")
    md.append("| --- | --- |")
    for c in raw.columns:
        md.append(f"| `{c}` | {int((raw[c].str.strip() == '').sum())} |")
    md.append("")

    # ---------- 2. 标签 ----------
    exp_num = pd.to_numeric(raw["exp"], errors="coerce")
    raw["exp_num"] = exp_num
    bad_label = exp_num.isna()
    extreme = exp_num.abs() > EXP_ABS_LIMIT

    md.append("## 2. 标签列 `exp`")
    md.append("")
    md.append(f"- 转不成数字的行：**{int(bad_label.sum())}**")
    md.append(f"- 数值超出合理范围（|logD| > {EXP_ABS_LIMIT:g}）的行：**{int(extreme.sum())}**")
    md.append("")
    if int((bad_label | extreme).sum()) > 0:
        md.append("| 行号 | CMPD_CHEMBLID | exp 原值 |")
        md.append("| --- | --- | --- |")
        for i in raw.index[bad_label | extreme]:
            md.append(f"| {i + 2} | `{raw.at[i, 'CMPD_CHEMBLID']}` | `{raw.at[i, 'exp']}` |")
        md.append("")

    valid = exp_num[exp_num.notna() & (exp_num.abs() <= EXP_ABS_LIMIT)]
    md.append("剔除异常值后的标签分布：")
    md.append("")
    md.append("| 指标 | 值 |")
    md.append("| --- | --- |")
    md.append(f"| 样本数 | {len(valid)} |")
    md.append(f"| 最小值 | {num(valid.min())} |")
    md.append(f"| 25% 分位 | {num(valid.quantile(0.25))} |")
    md.append(f"| 中位数 | {num(valid.median())} |")
    md.append(f"| 75% 分位 | {num(valid.quantile(0.75))} |")
    md.append(f"| 最大值 | {num(valid.max())} |")
    md.append(f"| 均值 | {num(valid.mean())} |")
    md.append(f"| 标准差 | {num(valid.std())} |")
    md.append("")

    # ---------- 3. ID ----------
    dup_id_rows = int(raw["CMPD_CHEMBLID"].duplicated().sum())
    nonstd = [x for x in raw["CMPD_CHEMBLID"] if not ID_PATTERN.match(str(x))]

    md.append("## 3. ID 列 `CMPD_CHEMBLID`")
    md.append("")
    md.append(f"- 重复的 ID 行数：**{dup_id_rows}**")
    md.append(f"- 不符合 `CHEMBL+数字` 格式的 ID：**{len(nonstd)}**")
    if nonstd:
        md.append("")
        md.append("  " + "、".join(f"`{x}`" for x in nonstd))
        md.append("")
        md.append("> 其中 `ljiohohoihoa` 的分子结构和标签都正常，只是 ID 本身是乱码，"
                  "不影响训练，不必剔除。")
        md.append("")
    md.append("> ID 不能当唯一键用，去重必须按分子结构（canonical SMILES）来判。")
    md.append("")

    # ---------- 4. SMILES ----------
    smiles = raw["smiles"]
    empty_smiles = smiles.str.strip() == ""
    too_long = smiles.str.len() > MAX_SMILES_LEN

    # 超长 SMILES 必须在解析前拦掉：RDKit 处理几千个原子的链会递归爆栈，
    # 整个进程直接崩（0xC00000FD），连异常都抓不到。
    canon, mol_ok, skipped_long = [], [], []
    for s in smiles:
        if not s.strip():
            canon.append(None)
            mol_ok.append(False)
            skipped_long.append(False)
            continue
        if len(s) > MAX_SMILES_LEN:
            canon.append(None)
            mol_ok.append(False)
            skipped_long.append(True)
            continue
        skipped_long.append(False)
        if not HAS_RDKIT:
            canon.append(None)
            mol_ok.append(True)
            continue
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            canon.append(None)
            mol_ok.append(False)
        else:
            canon.append(Chem.MolToSmiles(mol))
            mol_ok.append(True)
    raw["canon"] = canon
    raw["mol_ok"] = mol_ok
    raw["skipped_long"] = skipped_long

    parse_fail = (~raw["mol_ok"]) & (~empty_smiles) & (~raw["skipped_long"])

    md.append("## 4. 结构列 `smiles`")
    md.append("")
    md.append(f"- 空 SMILES：**{int(empty_smiles.sum())}**")
    if HAS_RDKIT:
        md.append(f"- RDKit 无法解析：**{int(parse_fail.sum())}**")
    else:
        md.append("- 未安装 RDKit，跳过解析检查")
    md.append(f"- 长度超过 {MAX_SMILES_LEN} 字符：**{int(too_long.sum())}**")
    md.append("")
    md.append("SMILES 长度分布（字符数）：")
    md.append("")
    md.append("| 指标 | 值 |")
    md.append("| --- | --- |")
    L = smiles.str.len()
    md.append(f"| 最短 | {int(L.min())} |")
    md.append(f"| 中位数 | {num(L.median(), 0)} |")
    md.append(f"| 最长 | {int(L.max())} |")
    md.append("")

    bad_rows = []
    for i in raw.index:
        why = []
        if bad_label[i]:
            why.append("标签为空或非数值")
        if extreme[i]:
            why.append(f"标签异常（{raw.at[i, 'exp']}）")
        if empty_smiles[i]:
            why.append("SMILES 为空")
        if too_long[i]:
            why.append(f"SMILES 过长（{len(raw.at[i, 'smiles'])} 字符）")
        if parse_fail[i]:
            why.append("SMILES 无法解析")
        if why:
            exp_raw = raw.at[i, "exp"].strip()
            bad_rows.append((i + 2, raw.at[i, "CMPD_CHEMBLID"],
                             exp_raw if exp_raw else "（空）",
                             raw.at[i, "smiles"][:40], "；".join(why)))

    if bad_rows:
        md.append("需要剔除的异常行：")
        md.append("")
        md.append("| 行号 | CMPD_CHEMBLID | exp | SMILES（截断） | 原因 |")
        md.append("| --- | --- | --- | --- | --- |")
        for r in bad_rows:
            md.append(f"| {r[0]} | `{r[1]}` | `{r[2]}` | `{r[3]}` | {r[4]} |")
        md.append("")

    # ---------- 5. 重复 ----------
    exact_dup = int(raw.duplicated(subset=list(raw.columns[:3])).sum())
    ok_rows = raw[raw["canon"].notna()]
    dup_canon_groups = int(ok_rows["canon"].duplicated().sum())

    md.append("## 5. 重复情况")
    md.append("")
    md.append(f"- 三列完全相同的重复行：**{exact_dup}**")
    md.append(f"- 按 canonical SMILES 重复的行：**{dup_canon_groups}**")
    md.append("")

    # 只看标签有效的行：异常标签（空 / 999999）本身不算"矛盾"
    valid_rows = raw[raw["canon"].notna() & raw["exp_num"].notna()
                     & (raw["exp_num"].abs() <= EXP_ABS_LIMIT)]
    conflict = (valid_rows.groupby("canon")["exp_num"].nunique() > 1).sum()
    md.append(f"- 同一结构但标签互相矛盾的结构数：**{int(conflict)}**")
    md.append("")
    md.append("> 只统计标签有效的行。`CHEMBL_BAD_EXP_EXTREME`（999999.0）虽然和某个正常分子"
              "结构相同，但那是坏标签，不算矛盾。")
    md.append("")
    md.append("> 这些重复必须在划分数据集之前处理掉，否则同一分子会同时出现在训练集"
              "和测试集，RMSE 会被人为压低。")
    md.append("")

    # ---------- 6. 清洗预估 ----------
    bad_mask = bad_label | extreme | empty_smiles | too_long | parse_fail
    after_drop = raw[~bad_mask]
    after_dedup = after_drop.drop_duplicates(subset="canon", keep="first")

    md.append("## 6. 清洗后样本量预估")
    md.append("")
    md.append("| 阶段 | 行数 |")
    md.append("| --- | --- |")
    md.append(f"| 原始 | {n} |")
    md.append(f"| 剔除异常行后 | {len(after_drop)}（剔除 {int(bad_mask.sum())}） |")
    md.append(f"| 按 canonical SMILES 去重后 | {len(after_dedup)} |")
    md.append("")

    md.append("## 7. 结论")
    md.append("")
    md.append(f"1. 数据不能直接拿来训练：共 **{int(bad_mask.sum())}** 行存在标签或结构问题，"
              "必须先剔除。")
    md.append(f"2. 有 **{exact_dup}** 行完全重复、**{dup_canon_groups}** 行结构重复，"
              "必须去重，否则验证结果不可信。")
    md.append(f"3. 清洗去重后可用样本约 **{len(after_dedup)}** 条。")
    md.append(f"4. 标签量级：中位数 {num(valid.median(), 2)}，标准差 {num(valid.std(), 2)}，"
              "后续 RMSE 应与这个量级对比才有意义。")
    md.append("")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(md), encoding="utf-8")

    # ---------- 控制台摘要 ----------
    print(f"数据文件：{CSV_PATH}")
    print(f"总行数  ：{n}")
    print(f"异常行  ：{int(bad_mask.sum())}  （明细见报告）")
    print(f"完全重复：{exact_dup} 行；结构重复：{dup_canon_groups} 行")
    print(f"清洗预估：{len(after_drop)} 行 → 去重后 {len(after_dedup)} 行")
    print(f"标签范围：{num(valid.min(), 2)} ~ {num(valid.max(), 2)}，中位数 {num(valid.median(), 2)}")
    print(f"报告已写入：{REPORT_PATH}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Lipophilicity.csv 数据质量检查与清洗脚本。

用法::

    python clean_lipophilicity.py                 # 读取 ./Lipophilicity.csv
    python clean_lipophilicity.py --input Lipophilicity.csv --outdir .

输出（默认写到脚本所在目录）::

    Lipophilicity_clean.csv          清洗后的数据集，列与原文件完全一致
                                     (CMPD_CHEMBLID, exp, smiles)
    Lipophilicity_clean_details.csv  清洗后数据 + 质检列
                                     (canonical_smiles, heavy_atoms, mol_weight, inchikey, qc_note)
    Lipophilicity_removed_rows.csv   被删除的样本及删除原因
    data_qc_summary.json             各类异常的数量统计

所有删除都是“显式规则 + 显式原因”，不做任何静默丢样本的处理。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

import pandas as pd

# 优先用当前解释器（如 conda "trial" 环境）自带的 RDKit；
# 只有环境里没有 RDKit 时才回退到工作区 .pylibs 的便携副本。
try:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Descriptors
except ImportError:
    sys.path.insert(0, os.path.join(HERE, ".pylibs"))
    try:
        from rdkit import Chem, RDLogger
        from rdkit.Chem import Descriptors
    except ImportError:  # pragma: no cover
        sys.exit("需要 RDKit：pip install rdkit")

RDLogger.DisableLog("rdApp.*")

# ---------------------------------------------------------------- 规则阈值
ID_PATTERN = re.compile(r"^CHEMBL\d{1,7}$")  # ChEMBL 编号格式（本数据集真实编号最多 7 位数字）
TARGET_MIN, TARGET_MAX = -3.0, 6.0          # logD(7.4) 的合理物理范围，超出记为异常
MAX_HEAVY_ATOMS = 150                       # 重原子数上限（超出视为非类药超大分子）
MAX_MOL_WEIGHT = 2000.0                     # 分子量上限 (Da)，只为拦截聚合物级别的异常分子
LABEL_CONFLICT_TOL = 0.3                    # 同一分子重复记录的 exp 差异容忍度（log 单位）

REASON_ZH = {
    "missing_smiles": "SMILES 为空",
    "missing_target": "exp 标签缺失",
    "invalid_target": "exp 不是数值",
    "invalid_smiles": "RDKit 无法解析（非法 SMILES）",
    "target_out_of_range": "exp 超出 logD 合理范围",
    "size_outlier": "分子尺寸异常（重原子/分子量过大）",
    "duplicate_row": "整行完全重复",
    "duplicate_molecule": "同一分子（规范化后相同）重复",
    "conflicting_label": "同一分子标签互相矛盾",
    "kept": "保留",
}


def parse_target(value: str):
    """解析 exp 列。返回 (float|None, 状态, 备注)。"""
    s = "" if value is None else str(value).strip()
    if s == "":
        return None, "missing_target", "空字符串"
    try:
        v = float(s)
    except ValueError:
        return None, "invalid_target", "非数值文本 %r" % s
    note = ""
    if s.startswith(".") or s.startswith("-."):
        note = "compact"          # ".05" 这类省略整数位的写法，数值本身合法
    return v, "ok", note


def smiles_info(smi: str):
    """用 RDKit 解析 SMILES，返回描述分子质量/身份的字典。"""
    info = {
        "mol": None, "canonical": None, "smiles_out": smi, "heavy_atoms": None,
        "mol_weight": None, "n_fragments": None, "status": "ok", "note": "",
    }
    if smi == "":
        info["status"] = "missing_smiles"
        return info
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        info["status"] = "invalid_smiles"
        return info

    info["n_fragments"] = len(Chem.GetMolFrags(mol))
    info["heavy_atoms"] = mol.GetNumHeavyAtoms()
    info["mol_weight"] = round(Descriptors.MolWt(mol), 2)

    # 含盐/水合物的多组分体系：只保留最大片段（去盐）后再规范化
    if info["n_fragments"] > 1:
        frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
        mol = max(frags, key=lambda m: m.GetNumHeavyAtoms())
        info["smiles_out"] = Chem.MolToSmiles(mol)
        info["note"] = "多组分 SMILES（%d 个片段），已去除盐/溶剂保留最大片段" % info["n_fragments"]

    # 超大分子跳过规范化：RDKit 对数千原子的链做 canonical ranking 会栈溢出
    if info["heavy_atoms"] > MAX_HEAVY_ATOMS:
        return info

    info["mol"] = mol
    try:
        info["canonical"] = Chem.MolToSmiles(mol)
    except Exception as exc:  # pragma: no cover
        info["note"] = (info["note"] + "; 规范化失败: %s" % exc).strip("; ")
    return info


def load_raw(path: str) -> pd.DataFrame:
    # keep_default_na=False：让缺失值表现为空字符串，避免 "NA"/"N/A" 之类被静默当成 NaN
    df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    expected = ["CMPD_CHEMBLID", "exp", "smiles"]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        sys.exit("输入文件缺少列: %s（实际列: %s）" % (missing, list(df.columns)))
    for col in expected:
        df[col] = df[col].astype(str).str.strip()
    df["src_line"] = df.index + 2  # 对应文本编辑器里的行号（第 1 行是表头）
    return df[expected + ["src_line"]]


def main() -> int:
    ap = argparse.ArgumentParser(description="Lipophilicity 数据集质量检查与清洗")
    ap.add_argument("--input", default=os.path.join(HERE, "Lipophilicity.csv"))
    ap.add_argument("--outdir", default=HERE)
    args = ap.parse_args()

    df = load_raw(args.input)
    raw_n = len(df)

    # ------------------------------------------------ 1. 逐行质检
    records = []
    compact_exp = 0
    for r in df.itertuples(index=False):
        exp_val, exp_status, exp_note = parse_target(r.exp)
        info = smiles_info(r.smiles)
        id_ok = bool(ID_PATTERN.match(r.CMPD_CHEMBLID))
        notes = []
        if not id_ok:
            notes.append("化合物 ID 不符合 ChEMBL 编号格式")
        if exp_note == "compact":
            compact_exp += 1
        if info["note"]:
            notes.append(info["note"])
        records.append({
            "src_line": r.src_line,
            "CMPD_CHEMBLID": r.CMPD_CHEMBLID,
            "id_ok": id_ok,
            "exp_raw": r.exp,
            "smiles": r.smiles,
            "smiles_out": info["smiles_out"],
            "exp": exp_val,
            "exp_status": exp_status,
            "smiles_status": info["status"],
            "canonical_smiles": info["canonical"],
            "heavy_atoms": info["heavy_atoms"],
            "mol_weight": info["mol_weight"],
            "n_fragments": info["n_fragments"],
            "note": "; ".join(notes),
        })
    rec = pd.DataFrame(records)

    # ------------------------------------------------ 2. 规则判定（按优先级，只记录第一条命中的原因）
    reasons = []
    for row in rec.itertuples(index=False):
        if row.smiles_status == "missing_smiles":
            reasons.append("missing_smiles")
        elif row.exp_status == "missing_target":
            reasons.append("missing_target")
        elif row.smiles_status == "invalid_smiles":
            reasons.append("invalid_smiles")
        elif row.exp_status == "invalid_target":
            reasons.append("invalid_target")
        elif row.exp is not None and not (TARGET_MIN <= row.exp <= TARGET_MAX):
            reasons.append("target_out_of_range")
        elif row.heavy_atoms is not None and (
            row.heavy_atoms > MAX_HEAVY_ATOMS or row.mol_weight > MAX_MOL_WEIGHT
        ):
            reasons.append("size_outlier")
        else:
            reasons.append("kept")
    rec["reason"] = reasons

    # ------------------------------------------------ 3. 完全重复行（三列全相同）
    dup_mask = rec.duplicated(subset=["CMPD_CHEMBLID", "exp_raw", "smiles"], keep="first")
    rec.loc[dup_mask & (rec["reason"] == "kept"), "reason"] = "duplicate_row"

    # ------------------------------------------------ 4. 同一分子（规范化后）重复 + 标签冲突
    alive = rec["reason"] == "kept"
    groups = rec[alive].groupby("canonical_smiles", dropna=True)
    for _, idx in groups.groups.items():
        sub = rec.loc[idx]
        labels = sub["exp"].dropna()
        conflict = len(labels) > 1 and (labels.max() - labels.min()) > LABEL_CONFLICT_TOL
        if conflict:
            # 同分子多标签互相矛盾：保留唯一一条 ID 合法、标签在合理范围内的记录；
            # 若无法唯一确定（0 条或多条合法记录），整组视为不可靠全部丢弃
            good = sub[sub["CMPD_CHEMBLID"].str.match(ID_PATTERN)]
            if len(good) == 1:
                keep = good.index[0]
                rec.loc[sub.index.difference([keep]), "reason"] = "conflicting_label"
            else:
                rec.loc[sub.index, "reason"] = "conflicting_label"
        elif len(sub) > 1:
            rec.loc[sub.index[1:], "reason"] = "duplicate_molecule"

    # ------------------------------------------------ 5. 输出
    kept = rec[rec["reason"] == "kept"].copy()
    removed = rec[rec["reason"] != "kept"].copy()

    clean = kept[["CMPD_CHEMBLID", "exp", "smiles_out"]].rename(columns={"smiles_out": "smiles"}).copy()
    clean["exp"] = clean["exp"].map(lambda v: ("%.2f" % v).rstrip("0").rstrip("."))
    clean_path = os.path.join(args.outdir, "Lipophilicity_clean.csv")
    clean.to_csv(clean_path, index=False)

    details = kept.drop(columns=["exp_raw", "exp_status", "smiles_status", "reason"])
    details["qc_note"] = kept["note"].fillna("")
    details = details[["CMPD_CHEMBLID", "exp", "smiles_out", "smiles", "canonical_smiles",
                       "heavy_atoms", "mol_weight", "n_fragments", "src_line", "qc_note"]]
    details = details.rename(columns={"smiles_out": "smiles_standardized", "smiles": "smiles_original"})
    details.to_csv(os.path.join(args.outdir, "Lipophilicity_clean_details.csv"), index=False)

    removed_out = removed[["src_line", "CMPD_CHEMBLID", "exp_raw", "smiles", "reason", "note"]].copy()
    removed_out["reason_zh"] = removed_out["reason"].map(REASON_ZH)
    removed_out.to_csv(os.path.join(args.outdir, "Lipophilicity_removed_rows.csv"), index=False)

    # ------------------------------------------------ 6. 统计
    counts = rec["reason"].value_counts().to_dict()
    summary = {
        "input_rows": raw_n,
        "kept_rows": int(len(kept)),
        "removed_rows": int(len(removed)),
        "removed_by_reason": {k: int(v) for k, v in counts.items() if k != "kept"},
        "removed_by_reason_zh": {REASON_ZH.get(k, k): int(v) for k, v in counts.items() if k != "kept"},
        "duplicate_row_extra": int((rec["reason"] == "duplicate_row").sum()),
        "exp_values_without_leading_zero": int(compact_exp),
        "target_after": {
            "min": round(float(kept["exp"].min()), 3),
            "max": round(float(kept["exp"].max()), 3),
            "mean": round(float(kept["exp"].mean()), 3),
            "std": round(float(kept["exp"].std()), 3),
            "median": round(float(kept["exp"].median()), 3),
        },
        "molecule_after": {
            "heavy_atoms_max": int(kept["heavy_atoms"].max()),
            "mol_weight_max": round(float(kept["mol_weight"].max()), 1),
            "multi_fragment_desalted": int((kept["n_fragments"] > 1).sum()),
        },
    }
    with open(os.path.join(args.outdir, "data_qc_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    # ------------------------------------------------ 7. 控制台报告
    print("=" * 68)
    print("Lipophilicity 数据质检报告")
    print("=" * 68)
    print("原始样本数: %d" % raw_n)
    print("保留样本数: %d (%.1f%%)" % (len(kept), 100.0 * len(kept) / raw_n))
    print("删除样本数: %d" % len(removed))
    print("-" * 68)
    for k, v in counts.items():
        if k == "kept":
            continue
        print("  %-22s %-28s %4d" % (k, REASON_ZH.get(k, k), v))
    print("-" * 68)
    print("清洗后 exp: min=%.2f max=%.2f mean=%.2f std=%.2f"
          % (summary["target_after"]["min"], summary["target_after"]["max"],
             summary["target_after"]["mean"], summary["target_after"]["std"]))
    print("exp 省略整数位的写法（如 '.05'）: %d 条，已统一转为标准浮点文本" % compact_exp)
    print("清洗后最大重原子数=%d, 最大分子量=%.1f, 去盐处理的多组分分子=%d"
          % (summary["molecule_after"]["heavy_atoms_max"],
             summary["molecule_after"]["mol_weight_max"],
             summary["molecule_after"]["multi_fragment_desalted"]))
    print("-" * 68)
    print("输出: %s" % clean_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

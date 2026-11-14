"""make_demo_test.py — 生成演示用的 test.csv

老师会另给 test.csv，这个脚本只是让你本地能先跑通接口：
  - 15 条真实样本（从 data/clean.csv 抽，标签真实，能算出有意义的 RMSE）
  - 5 条故意写坏的行（空 SMILES、乱码、括号不闭合、超长链、标签是文字）
    用来证明接口遇到坏行不会崩

运行：python src/make_demo_test.py
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CLEAN_CSV = ROOT / "data" / "clean.csv"
OUT_CSV = ROOT / "test.csv"

N_REAL = 15
SEED = 42


def main() -> None:
    if not CLEAN_CSV.exists():
        raise SystemExit("找不到 data/clean.csv，先运行 python src/02_clean.py")

    clean = pd.read_csv(CLEAN_CSV)
    real = clean.sample(n=N_REAL, random_state=SEED)[["CMPD_CHEMBLID", "exp", "smiles"]]

    good_smiles = real.iloc[0]["smiles"]
    broken = pd.DataFrame([
        ("DEMO_BLANK_SMILES", "2.10", ""),
        ("DEMO_BAD_SMILES", "1.50", "not_a_smiles"),
        ("DEMO_BAD_PAREN", "2.00", "C1(CC"),
        ("DEMO_TOO_LONG", "3.00", "C" * 400),
        ("DEMO_BAD_EXP", "definitely_wrong", good_smiles),
    ], columns=["CMPD_CHEMBLID", "exp", "smiles"])

    out = pd.concat([real, broken], ignore_index=True)
    out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    print(f"已生成 {OUT_CSV}")
    print(f"  真实样本 {len(real)} 条 + 故意写坏 {len(broken)} 条 = {len(out)} 行")


if __name__ == "__main__":
    main()

"""07_predict_test.py — 一条命令完成预测与打分

用法：
    python src/07_predict_test.py --input test.csv --output predictions.csv

输入 test.csv 必须有三列：CMPD_CHEMBLID, exp, smiles（与 Lipophilicity.csv 同字段）。
  - exp 可以缺失或非法：照样预测，但不参与打分
  - smiles 为空 / 非法 / 超长：该行不预测，单独列进报告的失败明细，**不会让程序崩**

输出：
  predictions.csv            每行的预测值与状态
  reports/predict_report.md  行数统计、失败明细、RMSE/MAE/R²、误差最大的样本

特征一律调用 src/featurize.py（与训练时同一份代码），模型与特征集从
models/best_model_meta.json 读，保证预测时的列序与训练完全一致。
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import featurize as fz

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "best_model.joblib"
META_PATH = ROOT / "models" / "best_model_meta.json"

NEED_COLS = ["CMPD_CHEMBLID", "smiles"]
EXP_ABS_LIMIT = 10.0


def read_table(path: Path) -> pd.DataFrame:
    """先按 UTF-8 读，失败再按 GBK 读（Excel 另存的 CSV 常是 GBK）。"""
    for enc in ("utf-8-sig", "gbk"):
        try:
            return pd.read_csv(path, dtype=str, keep_default_na=False, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise SystemExit(f"读不了 {path}：编码既不是 UTF-8 也不是 GBK")


def main() -> None:
    ap = argparse.ArgumentParser(description="用训练好的模型预测 test.csv 并打分")
    ap.add_argument("--input", default="test.csv", help="待预测的 CSV")
    ap.add_argument("--output", default="predictions.csv", help="预测结果 CSV")
    ap.add_argument("--report", default="reports/predict_report.md", help="报告输出路径")
    args = ap.parse_args()

    if not MODEL_PATH.exists():
        raise SystemExit("找不到模型，先运行：python src/06_train_improved.py")
    inp = Path(args.input)
    if not inp.exists():
        raise SystemExit(f"找不到输入文件：{inp.resolve()}")

    model = joblib.load(MODEL_PATH)
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    feature_set = meta["feature_set"]

    df = read_table(inp)
    missing = [c for c in NEED_COLS if c not in df.columns]
    if missing:
        raise SystemExit(f"输入缺少必需列：{missing}（需要 {NEED_COLS}，exp 可有可无）")
    if "exp" not in df.columns:
        df["exp"] = ""

    n = len(df)

    # ---- 逐行判定能不能预测 ----
    mols = fz.mols_from_smiles(df["smiles"].tolist())
    status, note = [], []
    for i, smi in enumerate(df["smiles"]):
        if not smi.strip():
            status.append("failed"); note.append("SMILES 为空")
        elif len(smi) > fz.MAX_SMILES_LEN:
            status.append("failed"); note.append(f"SMILES 过长（{len(smi)} 字符）")
        elif mols[i] is None:
            status.append("failed"); note.append("SMILES 无法解析")
        else:
            status.append("ok"); note.append("")

    ok_mask = np.array([s == "ok" for s in status])
    ok_mols = [m for m, ok in zip(mols, ok_mask) if ok]

    # ---- 按训练时的特征集拼特征 ----
    X = np.zeros((len(ok_mols), 0), dtype=np.float32)
    if feature_set in ("desc", "desc+morgan"):
        X = np.hstack([X, fz.desc_frame(ok_mols).to_numpy(dtype=np.float32)])
    if feature_set in ("morgan", "desc+morgan"):
        X = np.hstack([X, fz.morgan_matrix(ok_mols).astype(np.float32)])

    preds = model.predict(X) if len(ok_mols) else np.array([])

    df["pred"] = np.nan
    df.loc[df.index[ok_mask], "pred"] = preds
    df["status"] = status
    df["note"] = note

    # ---- 打分：只算 预测成功 + 标签合法 的行 ----
    exp_num = pd.to_numeric(df["exp"], errors="coerce").to_numpy(dtype=float)
    scorable = ok_mask & ~np.isnan(exp_num) & (np.abs(exp_num) <= EXP_ABS_LIMIT)
    y_true = exp_num[scorable]
    y_pred = df["pred"].to_numpy(dtype=float)[scorable]

    m = {"n": int(scorable.sum())}
    if len(y_true) >= 2:
        m["rmse"] = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        m["mae"] = float(mean_absolute_error(y_true, y_pred))
        m["r2"] = float(r2_score(y_true, y_pred))
    else:
        m["rmse"] = m["mae"] = m["r2"] = None

    # ---- 写预测结果 ----
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out = df[["CMPD_CHEMBLID", "smiles", "exp", "pred", "status", "note"]].copy()
    out["pred"] = out["pred"].map(lambda v: "" if pd.isna(v) else f"{v:.3f}")
    out.to_csv(out_path, index=False, encoding="utf-8-sig")

    # ---- 失败与跳过的明细 ----
    failed = df[df["status"] == "failed"]
    bad_label = df[(df["status"] == "ok") & np.isnan(exp_num) & (df["exp"].str.strip() != "")]
    big_label = df[(df["status"] == "ok") & ~np.isnan(exp_num) & (np.abs(exp_num) > EXP_ABS_LIMIT)]

    # ---- 报告 ----
    md = []
    md.append("# 预测报告")
    md.append("")
    md.append(f"- 输入：`{inp}`（{n} 行）")
    md.append(f"- 模型：`{meta['model']}` + 特征集 `{meta['feature_set']}`"
              f"（训练时测试集 RMSE {meta['test_rmse']:.3f}）")
    md.append(f"- 输出：`{out_path}`")
    md.append("")
    md.append("## 1. 处理统计")
    md.append("")
    md.append("| 项目 | 行数 |")
    md.append("| --- | --- |")
    md.append(f"| 输入总行数 | {n} |")
    md.append(f"| 成功预测 | {int(ok_mask.sum())} |")
    md.append(f"| 结构无效（未预测） | {len(failed)} |")
    md.append(f"| 参与打分的行 | {m['n']} |")
    md.append("")
    md.append("## 2. 指标")
    md.append("")
    if m["rmse"] is None:
        md.append("参与打分的行少于 2 条，无法计算指标。")
    else:
        md.append("| 指标 | 值 |")
        md.append("| --- | --- |")
        md.append(f"| RMSE | **{m['rmse']:.3f}** |")
        md.append(f"| MAE | {m['mae']:.3f} |")
        md.append(f"| R² | {m['r2']:.3f} |")
        md.append(f"| 参与打分的行数 | {m['n']} |")
    md.append("")
    md.append("> 打分只包含“预测成功且标签是合法数字（|logD| ≤ 10）”的行；"
              "结构无效的行照样写在 predictions.csv 里，状态标记为 failed。")
    md.append("")
    md.append("## 3. 失败与跳过")
    md.append("")
    if len(failed) or len(bad_label) or len(big_label):
        md.append("| CMPD_CHEMBLID | 问题 |")
        md.append("| --- | --- |")
        for _, r in failed.iterrows():
            md.append(f"| `{r['CMPD_CHEMBLID']}` | {r['note']} |")
        for _, r in bad_label.iterrows():
            md.append(f"| `{r['CMPD_CHEMBLID']}` | 标签不是数字（`{r['exp']}`），已预测但不打分 |")
        for _, r in big_label.iterrows():
            md.append(f"| `{r['CMPD_CHEMBLID']}` | 标签越界（`{r['exp']}`），已预测但不打分 |")
    else:
        md.append("没有失败行，也没有被跳过的标签。")
    md.append("")
    md.append("## 4. 误差最大的 5 条")
    md.append("")
    if m["n"] >= 5:
        scored = df[scorable].copy()
        scored["abs_err"] = (scored["pred"] - exp_num[scorable]).abs()
        top = scored.nlargest(5, "abs_err")
        md.append("| CMPD_CHEMBLID | 真实 | 预测 | 绝对误差 |")
        md.append("| --- | --- | --- | --- |")
        for _, r in top.iterrows():
            md.append(f"| `{r['CMPD_CHEMBLID']}` | {float(r['exp']):.2f} | "
                      f"{r['pred']:.3f} | {r['abs_err']:.3f} |")
    else:
        md.append("参与打分的样本不足 5 条。")
    md.append("")
    md.append("## 5. 复现命令")
    md.append("")
    md.append("```bash")
    md.append(f"python src/07_predict_test.py --input {inp} --output {out_path}")
    md.append("```")
    md.append("")

    rep = Path(args.report)
    rep.parent.mkdir(parents=True, exist_ok=True)
    rep.write_text("\n".join(md), encoding="utf-8")

    print(f"输入      ：{inp}（{n} 行）")
    print(f"成功预测  ：{int(ok_mask.sum())} 行；结构无效 {len(failed)} 行（已写进结果，状态 failed）")
    if m["rmse"] is None:
        print("指标      ：参与打分的行不足 2 条，无法计算")
    else:
        print(f"打分      ：{m['n']} 行，RMSE {m['rmse']:.3f} / MAE {m['mae']:.3f} / R² {m['r2']:.3f}")
    print(f"预测结果  ：{out_path}")
    print(f"报告      ：{rep}")


if __name__ == "__main__":
    main()

"""05_train_baseline.py — 基线模型

用什么：12 个分子描述符 → 中位数填补 → 标准化 → Ridge 回归
为什么这么简单：先拿到一个"数字可信、流程闭环"的参照点。
后面任何更复杂的模型，都要跟这个基线比，比不过就说明没学到东西。

流程：
  1. 只用训练集调 alpha（看验证集 RMSE）
  2. 定下 alpha 后，用 训练集+验证集 重训，在测试集上出最终分数
  3. 同时跑一个"永远预测训练集均值"的哑模型，作为最低参照

输出：
  models/baseline_ridge.joblib   最终模型（管线整体保存，含填补和标准化）
  models/metrics_baseline.json   所有指标
  reports/baseline.md            基线报告
  reports/figs/baseline_pred.png 测试集预测 vs 真实

运行：python src/05_train_baseline.py
"""

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MODELS = ROOT / "models"
FIG_DIR = ROOT / "reports" / "figs"
REPORT_MD = ROOT / "reports" / "baseline.md"

ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0]


def metrics(y_true, y_pred):
    return {
        "n": int(len(y_true)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def make_pipeline(alpha):
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("ridge", Ridge(alpha=alpha)),
    ])


def main() -> None:
    X = np.load(DATA / "X_desc.npy")
    y = np.load(DATA / "y.npy")
    split = np.load(DATA / "split.npz")
    tr, va, te = split["train_idx"], split["val_idx"], split["test_idx"]

    X_tr, y_tr = X[tr], y[tr]
    X_va, y_va = X[va], y[va]
    X_te, y_te = X[te], y[te]

    # ---- 1. 在训练集上调 alpha，用验证集选 ----
    alpha_log = []
    best_alpha, best_val = None, np.inf
    for a in ALPHAS:
        m = make_pipeline(a).fit(X_tr, y_tr)
        v = np.sqrt(mean_squared_error(y_va, m.predict(X_va)))
        alpha_log.append((a, v))
        if v < best_val:
            best_alpha, best_val = a, v

    # ---- 2. 哑模型：永远预测训练集均值 ----
    dummy = DummyRegressor(strategy="mean").fit(X_tr, y_tr)
    dummy_val = metrics(y_va, dummy.predict(X_va))
    dummy_te = metrics(y_te, dummy.predict(X_te))
    dummy_tr = metrics(y_tr, dummy.predict(X_tr))

    # ---- 3. 定下 alpha，用 train+val 重训，在测试集出分 ----
    final = make_pipeline(best_alpha).fit(
        np.vstack([X_tr, X_va]), np.concatenate([y_tr, y_va])
    )
    ridge_tr = metrics(y_tr, make_pipeline(best_alpha).fit(X_tr, y_tr).predict(X_tr))
    ridge_va = metrics(y_va, make_pipeline(best_alpha).fit(X_tr, y_tr).predict(X_va))
    ridge_te = metrics(y_te, final.predict(X_te))

    # ---- 保存 ----
    MODELS.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(final, MODELS / "baseline_ridge.joblib")
    (MODELS / "metrics_baseline.json").write_text(
        json.dumps(
            {
                "model": f"Ridge(alpha={best_alpha}) on 12 descriptors",
                "best_alpha": best_alpha,
                "alpha_search": [{"alpha": a, "val_rmse": v} for a, v in alpha_log],
                "dummy": {"train": dummy_tr, "val": dummy_val, "test": dummy_te},
                "ridge": {"train": ridge_tr, "val": ridge_va, "test": ridge_te},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # ---- 画图：测试集预测 vs 真实 ----
    pred_te = final.predict(X_te)
    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    ax.scatter(y_te, pred_te, s=18, alpha=0.65, color="#1e3a5f", edgecolors="none")
    lim = [min(y_te.min(), pred_te.min()) - 0.3, max(y_te.max(), pred_te.max()) + 0.3]
    ax.plot(lim, lim, "--", color="#c0392b", linewidth=1, label="ideal y = x")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("True logD")
    ax.set_ylabel("Predicted logD")
    ax.set_title(
        f"Baseline (Ridge, 12 descriptors)\nTest RMSE = {ridge_te['rmse']:.3f}, "
        f"n = {ridge_te['n']}"
    )
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "baseline_pred.png", bbox_inches="tight")
    plt.close(fig)

    # ---- 报告 ----
    md = []
    md.append("# 基线模型报告")
    md.append("")
    md.append(f"- 模型：12 个描述符 → 中位数填补 → 标准化 → Ridge（alpha = {best_alpha}）")
    md.append(f"- 数据：训练 {len(tr)} / 验证 {len(va)} / 测试 {len(te)}（scaffold 划分）")
    md.append("")
    md.append("## 1. 指标")
    md.append("")
    md.append("| 模型 | 训练 RMSE | 验证 RMSE | 测试 RMSE | 测试 MAE | 测试 R² |")
    md.append("| --- | --- | --- | --- | --- | --- |")
    md.append(f"| 均值基线 | {dummy_tr['rmse']:.3f} | {dummy_val['rmse']:.3f} | "
              f"{dummy_te['rmse']:.3f} | {dummy_te['mae']:.3f} | {dummy_te['r2']:.3f} |")
    md.append(f"| Ridge | {ridge_tr['rmse']:.3f} | {ridge_va['rmse']:.3f} | "
              f"{ridge_te['rmse']:.3f} | {ridge_te['mae']:.3f} | {ridge_te['r2']:.3f} |")
    md.append("")
    md.append("> 均值基线的 R² 恒为 0、RMSE 等于标签标准差——它就是“什么也没学”的下限。"
              "任何模型至少要比它明显好，否则等于没做。")
    md.append("")
    md.append("## 2. alpha 选择（只在验证集上比）")
    md.append("")
    md.append("| alpha | 验证集 RMSE |")
    md.append("| --- | --- |")
    for a, v in alpha_log:
        md.append(f"| {a} | {v:.3f}{' ← 选中' if a == best_alpha else ''} |")
    md.append("")
    md.append("## 3. 结论")
    md.append("")
    gain = (dummy_te["rmse"] - ridge_te["rmse"]) / dummy_te["rmse"]
    md.append(f"1. 基线测试集 RMSE = **{ridge_te['rmse']:.3f}**，比“永远预测均值”好 "
              f"{gain:.1%}。")
    md.append(f"2. 12 个描述符只能表达整体物化性质，表达不了“这个子结构在哪个位置”，"
              f"这正是后面要加指纹特征的原因。")
    md.append(f"3. 这个数字就是后面所有改动的对照线：新模型测试集 RMSE 必须低于 "
              f"{ridge_te['rmse']:.3f} 才算有效改进。")
    md.append("")
    md.append("## 4. 图")
    md.append("")
    md.append("测试集预测 vs 真实：`reports/figs/baseline_pred.png`。点越贴近红色对角线越好。")
    md.append("")
    md.append("## 5. 输出文件")
    md.append("")
    md.append("| 文件 | 说明 |")
    md.append("| --- | --- |")
    md.append("| `models/baseline_ridge.joblib` | 完整管线（填补+标准化+Ridge），可直接 predict |")
    md.append("| `models/metrics_baseline.json` | 全部指标与 alpha 搜索记录 |")
    md.append("| `reports/figs/baseline_pred.png` | 测试集散点图 |")
    md.append("")

    REPORT_MD.write_text("\n".join(md), encoding="utf-8")

    print("alpha 搜索（验证集 RMSE）：")
    for a, v in alpha_log:
        print(f"  alpha={a:<6} {v:.4f}{'  ← 选中' if a == best_alpha else ''}")
    print("")
    print(f"均值基线 ：训练 {dummy_tr['rmse']:.3f} / 验证 {dummy_val['rmse']:.3f} / 测试 {dummy_te['rmse']:.3f}")
    print(f"Ridge    ：训练 {ridge_tr['rmse']:.3f} / 验证 {ridge_va['rmse']:.3f} / 测试 {ridge_te['rmse']:.3f}")
    print(f"测试集   ：MAE {ridge_te['mae']:.3f}，R² {ridge_te['r2']:.3f}，比均值基线好 {gain:.1%}")
    print(f"模型     ：{MODELS / 'baseline_ridge.joblib'}")
    print(f"报告     ：{REPORT_MD}")


if __name__ == "__main__":
    main()

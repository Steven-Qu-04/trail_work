"""06_train_improved.py — 改进模型

思路：基线只用了 12 个描述符，信息不够。这里把 Morgan 指纹（2048 位）
和描述符拼起来，再换更强的模型，看能不能明显压过基线的 0.976。

流程（严格照规矩来，不许偷看测试集）：
  1. 三套特征 × 三种模型 = 9 个组合，全部只在训练集上拟合
  2. 用验证集 RMSE 挑出最好的组合
  3. 用 训练集+验证集 重训这个组合，最后在测试集上出一次分
  4. 测试集只用来报最终成绩，不参与任何选择

输出：
  models/best_model.joblib        最优模型（含特征顺序信息，07 直接用）
  models/metrics_improved.json    9 个组合的验证分数 + 最终测试分数
  reports/improved.md             改进报告
  reports/figs/improved_pred.png  测试集预测 vs 真实

运行：python src/06_train_improved.py
"""

from pathlib import Path
import json
import time

import joblib
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import featurize as fz

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MODELS = ROOT / "models"
FIG_DIR = ROOT / "reports" / "figs"
REPORT_MD = ROOT / "reports" / "improved.md"

ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0]
SEED = 42


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def metrics(y_true, y_pred):
    return {
        "n": int(len(y_true)),
        "rmse": rmse(y_true, y_pred),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def build_model(name, alpha=None):
    impute = ("impute", SimpleImputer(strategy="median"))
    if name == "ridge":
        return Pipeline([impute, ("scale", StandardScaler()), ("model", Ridge(alpha=alpha))])
    if name == "rf":
        return Pipeline([
            impute,
            ("model", RandomForestRegressor(
                n_estimators=300, max_features="sqrt", min_samples_leaf=2,
                n_jobs=-1, random_state=SEED)),
        ])
    if name == "hgb":
        return Pipeline([
            impute,
            ("model", HistGradientBoostingRegressor(
                max_iter=250, learning_rate=0.06, max_leaf_nodes=31,
                min_samples_leaf=20, l2_regularization=0.1, random_state=SEED)),
        ])
    raise ValueError(name)


def main() -> None:
    X_desc = np.load(DATA / "X_desc.npy")
    X_morgan = np.load(DATA / "X_morgan.npy").astype(np.float32)
    y = np.load(DATA / "y.npy")
    split = np.load(DATA / "split.npz")
    tr, va, te = split["train_idx"], split["val_idx"], split["test_idx"]

    feature_sets = {
        "desc": X_desc,
        "morgan": X_morgan,
        "desc+morgan": np.hstack([X_desc, X_morgan]),
    }

    # ---- 1. 九个组合，只在训练集拟合，用验证集打分 ----
    results = []
    t0 = time.time()
    for fs_name, X in feature_sets.items():
        for model_name in ("ridge", "rf", "hgb"):
            t1 = time.time()
            if model_name == "ridge":
                best_a, best_v = None, np.inf
                for a in ALPHAS:
                    m = build_model("ridge", a).fit(X[tr], y[tr])
                    v = rmse(y[va], m.predict(X[va]))
                    if v < best_v:
                        best_a, best_v = a, v
                note = f"alpha={best_a}"
            else:
                m = build_model(model_name).fit(X[tr], y[tr])
                best_v = rmse(y[va], m.predict(X[va]))
                note = ""
            results.append({"feature_set": fs_name, "model": model_name,
                            "val_rmse": best_v, "note": note})
            print(f"  {fs_name:<12} {model_name:<6} 验证 RMSE {best_v:.4f} {note} "
                  f"({time.time() - t1:.1f}s)", flush=True)
    print(f"九组总耗时 {time.time() - t0:.1f}s", flush=True)

    best = min(results, key=lambda r: r["val_rmse"])
    X_best = feature_sets[best["feature_set"]]

    # ---- 2. 用 train+val 重训，测试集只出一次分 ----
    alpha = None
    if best["model"] == "ridge":
        alpha = float(best["note"].split("=")[1])
    final = build_model(best["model"], alpha).fit(
        np.vstack([X_best[tr], X_best[va]]), np.concatenate([y[tr], y[va]])
    )
    pred_te = final.predict(X_best[te])
    test_m = metrics(y[te], pred_te)

    # 基线成绩，用于对比
    baseline_rmse = None
    base_json = MODELS / "metrics_baseline.json"
    if base_json.exists():
        baseline_rmse = json.loads(base_json.read_text(encoding="utf-8"))["ridge"]["test"]["rmse"]

    # ---- 3. 特征贡献：整块打乱法 ----
    # HistGradientBoosting 没有 feature_importances_ 属性，所以换个办法：
    # 把某一块特征的行顺序随机打乱（破坏它与标签的对应关系），看验证集 RMSE 涨多少。
    # 涨得越多，说明这块特征越不可替代。用只见过训练集的模型，保证公平。
    m_val = build_model(best["model"], alpha).fit(X_best[tr], y[tr])
    X_va = X_best[va]
    base_val = rmse(y[va], m_val.predict(X_va))
    n_desc = X_desc.shape[1]
    blocks = []
    if best["feature_set"] in ("desc", "desc+morgan"):
        blocks.append(("描述符", slice(0, n_desc)))
    if best["feature_set"] in ("morgan", "desc+morgan"):
        blocks.append(("Morgan 指纹", slice(n_desc, X_best.shape[1])))

    rng = np.random.default_rng(SEED)
    block_imps = []
    for name, sl in blocks:
        scores = []
        for _ in range(5):
            Xp = X_va.copy()
            perm = rng.permutation(Xp.shape[0])
            Xp[:, sl] = Xp[perm, sl]
            scores.append(rmse(y[va], m_val.predict(Xp)))
        block_imps.append((name, float(np.mean(scores))))

    # ---- 4. 保存 ----
    MODELS.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(final, MODELS / "best_model.joblib")
    (MODELS / "best_model_meta.json").write_text(
        json.dumps({
            "feature_set": best["feature_set"],
            "model": best["model"],
            "alpha": alpha,
            "descriptors": fz.desc_names(),
            "morgan_radius": fz.MORGAN_RADIUS,
            "morgan_bits": fz.MORGAN_BITS,
            "n_features": int(X_best.shape[1]),
            "val_rmse": best["val_rmse"],
            "test_rmse": test_m["rmse"],
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (MODELS / "metrics_improved.json").write_text(
        json.dumps({"results": results, "best": best, "test": test_m,
                    "baseline_test_rmse": baseline_rmse,
                    "val_rmse_clean": base_val,
                    "block_importance": [
                        {"block": n, "val_rmse_after_shuffle": v} for n, v in block_imps
                    ]},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # ---- 5. 图 ----
    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    ax.scatter(y[te], pred_te, s=18, alpha=0.65, color="#1e3a5f", edgecolors="none")
    lim = [min(y[te].min(), pred_te.min()) - 0.3, max(y[te].max(), pred_te.max()) + 0.3]
    ax.plot(lim, lim, "--", color="#c0392b", linewidth=1, label="ideal y = x")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("True logD"); ax.set_ylabel("Predicted logD")
    ax.set_title(f"Improved: {best['model']} on {best['feature_set']}\n"
                 f"Test RMSE = {test_m['rmse']:.3f}, n = {test_m['n']}")
    ax.legend(loc="upper left", fontsize=9); ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "improved_pred.png", bbox_inches="tight")
    plt.close(fig)

    # ---- 6. 报告 ----
    md = []
    md.append("# 改进模型报告")
    md.append("")
    md.append(f"- 搜索空间：{len(feature_sets)} 套特征 × 3 种模型 = {len(results)} 个组合")
    md.append(f"- 选择依据：**只看验证集 RMSE**，测试集不参与选择")
    md.append("")
    md.append("## 1. 九个组合的验证集成绩")
    md.append("")
    md.append("| 特征 | 特征数 | 模型 | 验证 RMSE | 备注 |")
    md.append("| --- | --- | --- | --- | --- |")
    dims = {"desc": X_desc.shape[1], "morgan": X_morgan.shape[1],
            "desc+morgan": X_desc.shape[1] + X_morgan.shape[1]}
    for r in results:
        md.append(f"| {r['feature_set']} | {dims[r['feature_set']]} | {r['model']} | "
                  f"{r['val_rmse']:.4f}{' ← 选中' if r is best else ''} | {r['note']} |")
    md.append("")
    md.append("## 2. 最优模型在测试集上的成绩")
    md.append("")
    md.append("| 模型 | 测试 RMSE | 测试 MAE | 测试 R² |")
    md.append("| --- | --- | --- | --- |")
    md.append(f"| 基线（描述符+Ridge） | {baseline_rmse:.4f} | — | — |"
              if baseline_rmse else "| 基线 | — | — | — |")
    md.append(f"| **改进（{best['feature_set']} + {best['model']}）** | "
              f"**{test_m['rmse']:.4f}** | {test_m['mae']:.4f} | {test_m['r2']:.4f} |")
    md.append("")
    if baseline_rmse:
        delta = (baseline_rmse - test_m["rmse"]) / baseline_rmse
        md.append(f"相对基线改善 **{delta:.1%}**（{baseline_rmse:.4f} → {test_m['rmse']:.4f}）。")
        md.append("")
    md.append("## 3. 特征贡献（整块打乱法）")
    md.append("")
    md.append("| 扰动方式 | 验证 RMSE | 相对基准上升 |")
    md.append("| --- | --- | --- |")
    md.append(f"| 不扰动 | {base_val:.4f} | — |")
    for name, v in block_imps:
        md.append(f"| 打乱 {name} | {v:.4f} | +{v - base_val:.4f} |")
    md.append("")
    md.append("> HistGradientBoosting 不提供 `feature_importances_`，所以用“把整块特征的行"
              "随机打乱、破坏它和标签的对应关系，看验证集 RMSE 涨多少”来衡量贡献。"
              "涨得越多，说明这块特征越不可替代。")
    md.append("")
    md.append("## 4. 误差结构：按真实值分箱")
    md.append("")
    md.append("| 真实 logD 区间 | 样本数 | 真实均值 | 预测均值 | 偏差 |")
    md.append("| --- | --- | --- | --- | --- |")
    for lo, hi, name in [(-99, -0.5, "< -0.5"), (-0.5, 1.0, "-0.5 ~ 1.0"),
                         (1.0, 2.5, "1.0 ~ 2.5"), (2.5, 3.5, "2.5 ~ 3.5"), (3.5, 99, "> 3.5")]:
        sel = (y[te] >= lo) & (y[te] < hi)
        if sel.sum():
            md.append(f"| {name} | {int(sel.sum())} | {y[te][sel].mean():.2f} | "
                      f"{pred_te[sel].mean():.2f} | "
                      f"{pred_te[sel].mean() - y[te][sel].mean():+.2f} |")
    slope = float(np.polyfit(y[te], pred_te, 1)[0])
    corr = float(np.corrcoef(y[te], pred_te)[0, 1])
    md.append("")
    md.append(f"预测对真实的回归斜率 **{slope:.3f}**（理想为 1），相关系数 {corr:.3f}；"
              f"预测范围 {pred_te.min():.2f} ~ {pred_te.max():.2f}，"
              f"真实范围 {y[te].min():.2f} ~ {y[te].max():.2f}。"
              "两端都向中间收缩：低值被系统高估、高值被系统低估，"
              "说明树模型对训练集中稀少的极端样本不敢外推。")
    md.append("")
    md.append("## 5. 结论")
    md.append("")
    desc_only = min(r["val_rmse"] for r in results if r["feature_set"] == "desc")
    bmap = {n: v for n, v in block_imps}
    md.append(f"1. 加指纹是主要增益：单用描述符验证 RMSE {desc_only:.3f}，"
              f"加上指纹后降到 {best['val_rmse']:.3f}"
              f"（-{desc_only - best['val_rmse']:.3f}）。")
    if "描述符" in bmap and "Morgan 指纹" in bmap:
        md.append(f"2. 打乱实验显示：描述符块 +{bmap['描述符'] - base_val:.3f}、"
                  f"指纹块 +{bmap['Morgan 指纹'] - base_val:.3f}。"
                  "12 个描述符信息更集中、更不可替代；指纹含大量冗余位，但提供了"
                  "描述符给不出的局部结构信息。两者互补，缺一不可。")
    if baseline_rmse:
        gain_txt = f"比基线改善 {(baseline_rmse - test_m['rmse']) / baseline_rmse:.1%}"
    else:
        gain_txt = "基线成绩缺失"
    md.append(f"3. 最终测试 RMSE **{test_m['rmse']:.3f}**，R² {test_m['r2']:.3f}，{gain_txt}。")
    md.append("4. 测试集只评了一次，没有任何调参动作，这个分数可以直接写进报告。")
    md.append("")
    md.append("## 6. 输出文件")
    md.append("")
    md.append("| 文件 | 说明 |")
    md.append("| --- | --- |")
    md.append("| `models/best_model.joblib` | 最优管线，07 直接加载 |")
    md.append("| `models/best_model_meta.json` | 特征集与参数，保证预测时特征顺序一致 |")
    md.append("| `models/metrics_improved.json` | 全部 9 个组合的分数 |")
    md.append("| `reports/figs/improved_pred.png` | 测试集散点图 |")
    md.append("")

    REPORT_MD.write_text("\n".join(md), encoding="utf-8")

    print("")
    print(f"最优组合：{best['feature_set']} + {best['model']}（验证 RMSE {best['val_rmse']:.4f}）")
    print(f"测试集  ：RMSE {test_m['rmse']:.4f} / MAE {test_m['mae']:.4f} / R² {test_m['r2']:.4f}")
    if baseline_rmse:
        print(f"对比基线：{baseline_rmse:.4f} → {test_m['rmse']:.4f}"
              f"（改善 {(baseline_rmse - test_m['rmse']) / baseline_rmse:.1%}）")
    print(f"模型    ：{MODELS / 'best_model.joblib'}")
    print(f"报告    ：{REPORT_MD}")


if __name__ == "__main__":
    main()

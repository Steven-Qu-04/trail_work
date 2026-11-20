#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Lipophilicity logD 回归：RDKit 描述符 (F1) + 显式 MolLogD 特征 + Ridge 主模型

设计依据：design_plan.md（v0.1），本脚本只实现"主模型"这一条路径。

1. 数据
   - 输入 Lipophilicity_clean.csv（上一阶段清洗结果，3780 条，列：CMPD_CHEMBLID, exp, smiles）
   - 只读，不再做去重 / 去异常，避免引入新的泄漏

2. 划分
   - Bemis-Murcko scaffold split，按 8:1:1 划 train / val / test
   - 骨架组按"组内分子数从大到小"贪心分配；同尺寸组用 seed 打散，保证可复现
   - 划分结果落盘 splits.json

3. 特征（F1 + MolLogD）
   - rdkit.Chem.Descriptors._descList 全量描述符（217 列）
   - 显式追加 MolLogD 列：RDKit 没有 logD 描述符，按约定用 Crippen MolLogP 实现，
     并在相关性剪枝阶段强制保留，保证这条核心亲脂性特征一定进入模型
   - F1 预处理链（全部只在 train 上 fit）：
       ① Inf -> NaN，NaN 用训练集中位数填补
       ② 剔除常量列（VarianceThreshold=0）
       ③ 偏态处理（design_plan 中的可选分支，默认关闭，--skew 打开）
       ④ 标准化（StandardScaler）
       ⑤ 共线性剪枝（|Pearson r| > 0.95 的列对，保留与 y 相关性绝对值更高的一列）

4. 模型
   - Ridge（L2），alpha 在训练集内部用 GroupKFold（按骨架分组）做 5 折 CV 选择
   - 不使用测试集调参；测试集只在最后用一次

5. 评价
   - RMSE（主指标）、MAE、R²、Spearman ρ
   - 同时报告：train-only 模型在 val / test 上的表现，以及 train+val 重训后在 test 上的表现
   - 保存测试集逐分子的真实 exp、预测 logD 和残差到 test_predictions.csv

6. 推理接口
   - python test --predict test.csv
   - 同一列名（CMPD_CHEMBLID, exp, smiles；exp 可缺失），输出 predictions.csv；
     非法 SMILES 不静默丢弃，单独写 invalid_rows.csv 并给出数量

用法：
    python test                      # 训练 + 评估 + 保存模型
    python test --predict test.csv   # 用已保存模型对新数据推理并出报告
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger, rdBase
from rdkit.Chem import Crippen, Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy.stats import spearmanr
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.preprocessing import StandardScaler

# RDKit 对每个分子刷一条告警会把日志淹掉，这里统一关掉
RDLogger.DisableLog("rdApp.*")

DESC_LIST = Descriptors._descList
DESC_NAMES = [name for name, _ in DESC_LIST]
LOGD_NAME = "MolLogD"          # 显式追加的亲脂性特征（Crippen MolLogP 实现）
DEFAULT_SEED = 42
SPLIT_RATIOS = (0.8, 0.1, 0.1)
ALPHAS = np.logspace(-2, 2.5, 10)
CORR_THRESHOLD = 0.95
TARGET_COL = "exp"
SMILES_COL = "smiles"
ID_COL = "CMPD_CHEMBLID"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# 1. 数据读取
# --------------------------------------------------------------------------- #
def _read_csv_tolerant(path: Path) -> pd.DataFrame:
    """按常见编码依次尝试，避免在中文路径 / GBK 文件上抛出看不懂的 traceback。"""
    last_err = None
    for enc in ("utf-8", "utf-8-sig", "gbk"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError as err:  # 换下一个编码
            last_err = err
        except pd.errors.EmptyDataError as err:
            raise SystemExit(f"[error] file is empty or has no header: {path}") from err
        except pd.errors.ParserError as err:
            raise SystemExit(f"[error] failed to parse CSV: {path} -> {err}") from err
    raise SystemExit(f"[error] cannot detect file encoding: {path} -> {last_err}")


def load_dataset(path: Path, require_target: bool = True) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"[error] data file not found: {path}")
    if path.stat().st_size == 0:
        raise SystemExit(f"[error] data file is empty: {path}")

    df = _read_csv_tolerant(path)
    df.columns = [str(c).strip() for c in df.columns]

    need = [SMILES_COL] + ([TARGET_COL] if require_target else [])
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise SystemExit(
            f"[error] missing required column(s) {missing}; actual columns: {list(df.columns)}"
        )

    if require_target:
        df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
        bad_target = df[TARGET_COL].isna()
        if bad_target.any():
            raise SystemExit(
                f"[error] {int(bad_target.sum())} rows in {TARGET_COL} cannot be parsed as numbers"
            )
    elif TARGET_COL in df.columns:
        df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")

    df[SMILES_COL] = df[SMILES_COL].astype("string").fillna("").str.strip()
    if not (df[SMILES_COL].str.len() > 0).all():
        n_empty = int((df[SMILES_COL].str.len() == 0).sum())
        log(f"warning: {n_empty} rows have empty SMILES; they will be reported as invalid")
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 2. 描述符计算（带缓存）
# --------------------------------------------------------------------------- #
def _descriptor_chunk(smiles_chunk: list[str]) -> tuple[list[list[float]], list[bool]]:
    """工作进程：单个 chunk 的描述符 + MolLogD。返回 (数值行, 是否非法)。"""
    n_cols = len(DESC_LIST) + 1
    rows: list[list[float]] = []
    bad: list[bool] = []
    for smi in smiles_chunk:
        mol = Chem.MolFromSmiles(smi) if smi else None
        if mol is None:
            rows.append([np.nan] * n_cols)
            bad.append(True)
            continue
        vals: list[float] = []
        for _, fn in DESC_LIST:
            try:
                vals.append(float(fn(mol)))
            except Exception:  # 个别描述符对极端分子会抛异常，记 NaN 交给填补
                vals.append(np.nan)
        try:
            vals.append(float(Crippen.MolLogP(mol)))  # MolLogD 列
        except Exception:
            vals.append(np.nan)
        rows.append(vals)
        bad.append(False)
    return rows, bad


def _serial_descriptors(smiles: list[str], progress_every: int = 1000):
    rows: list[list[float]] = []
    bad: list[bool] = []
    for start in range(0, len(smiles), progress_every):
        r, b = _descriptor_chunk(smiles[start:start + progress_every])
        rows.extend(r)
        bad.extend(b)
        log(f"  descriptors {min(start + progress_every, len(smiles))}/{len(smiles)}")
    return rows, bad


def compute_descriptor_matrix(smiles: list[str], n_jobs: int = -1):
    """返回 (X, bad_mask)。X 的列 = DESC_NAMES + [MolLogD]。"""
    n = len(smiles)
    if n == 0:
        return np.empty((0, len(DESC_NAMES) + 1)), np.zeros(0, dtype=bool)

    workers = (os.cpu_count() or 1) if n_jobs in (-1, None) else int(n_jobs)
    workers = max(1, min(workers, 8, n))
    if workers > 1 and n >= 200:
        # 每个 worker 拿一段连续数据，避免切得太碎导致进程通信开销占主导
        chunk = int(np.ceil(n / workers))
        chunks = [smiles[i:i + chunk] for i in range(0, n, chunk)]
        try:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                results = list(pool.map(_descriptor_chunk, chunks))
            rows = [row for r, _ in results for row in r]
            bad = [flag for _, b in results for flag in b]
            log(f"  descriptors done on {workers} processes")
            return np.asarray(rows, dtype=float), np.asarray(bad, dtype=bool)
        except Exception as err:  # 多进程在个别平台/沙箱下会失败，退回串行
            log(f"  warning: parallel descriptor computation failed ({err}); falling back to serial")
    rows, bad = _serial_descriptors(smiles)
    return np.asarray(rows, dtype=float), np.asarray(bad, dtype=bool)


def feature_cache_key(smiles: list[str]) -> str:
    """缓存必须和数据绑定：SMILES 列表 + RDKit 版本 + 列数，任一变化就重算。"""
    h = hashlib.sha1()
    h.update("|".join(smiles).encode("utf-8"))
    h.update(f"#{rdBase.rdkitVersion}#{len(DESC_NAMES)}#{LOGD_NAME}".encode("utf-8"))
    return h.hexdigest()


def build_feature_matrix(smiles: list[str], cache_path: Path | None, n_jobs: int,
                         use_cache: bool = True) -> tuple[np.ndarray, np.ndarray, list[str]]:
    names = DESC_NAMES + [LOGD_NAME]
    key = feature_cache_key(smiles)

    if use_cache and cache_path is not None and cache_path.exists():
        try:
            cached = np.load(cache_path, allow_pickle=True)
            if str(cached["key"].item()) == key:
                log(f"  features loaded from cache: {cache_path.name}")
                return cached["X"], cached["bad"].astype(bool), list(cached["names"])
        except Exception as err:
            log(f"  warning: cache unusable ({err}); recomputing")

    started = time.time()
    log(f"  computing {len(DESC_NAMES) + 1} descriptors for {len(smiles)} molecules ...")
    X, bad = compute_descriptor_matrix(smiles, n_jobs=n_jobs)
    log(f"  descriptors computed in {time.time() - started:.1f}s")

    if use_cache and cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, X=X, bad=bad, names=np.asarray(names, dtype=object),
                            key=np.asarray(key))
        log(f"  features cached -> {cache_path}")
    return X, bad, names


# --------------------------------------------------------------------------- #
# 3. Bemis-Murcko scaffold split
# --------------------------------------------------------------------------- #
def compute_scaffolds(smiles: list[str]) -> list[str]:
    scaffolds = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi) if smi else None
        if mol is None:
            scaffolds.append("")
            continue
        scaffolds.append(MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False))
    return scaffolds


def scaffold_split(scaffolds: list[str], ratios=SPLIT_RATIOS,
                   seed: int = DEFAULT_SEED) -> dict[str, np.ndarray]:
    """按骨架组从大到小贪心填 train -> val -> test，保证同一骨架不跨集合。"""
    if abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError(f"ratios must sum to 1, got {ratios}")

    groups: dict[str, list[int]] = defaultdict(list)
    for idx, scaf in enumerate(scaffolds):
        groups[scaf].append(idx)

    rng = np.random.default_rng(seed)
    tie_break = {scaf: rng.random() for scaf in groups}   # 同尺寸组的顺序也可复现
    ordered = sorted(groups, key=lambda s: (-len(groups[s]), tie_break[s]))

    n = len(scaffolds)
    target_train = ratios[0] * n
    target_val = (ratios[0] + ratios[1]) * n

    assigned: dict[str, str] = {}
    n_train = n_val = 0
    for scaf in ordered:
        members = groups[scaf]
        if n_train < target_train:
            assigned[scaf] = "train"
            n_train += len(members)
        elif n_train + n_val < target_val:
            assigned[scaf] = "val"
            n_val += len(members)
        else:
            assigned[scaf] = "test"

    # 极端数据下可能某个集合为空：从上一个集合搬一个最小的组过来，保证三个集合都非空
    for src, dst in (("train", "val"), ("val", "test")):
        if not any(v == dst for v in assigned.values()):
            movable = [s for s, v in assigned.items() if v == src]
            if movable:
                assigned[min(movable, key=lambda s: len(groups[s]))] = dst

    train: list[int] = []
    val: list[int] = []
    test: list[int] = []
    for scaf, part in assigned.items():
        {"train": train, "val": val, "test": test}[part].extend(groups[scaf])

    return {
        "train": np.sort(np.asarray(train, dtype=int)),
        "val": np.sort(np.asarray(val, dtype=int)),
        "test": np.sort(np.asarray(test, dtype=int)),
    }


def split_report(splits: dict[str, np.ndarray], scaffolds: list[str],
                 y: np.ndarray) -> dict:
    scaf = np.asarray(scaffolds, dtype=object)
    tr, va, te = splits["train"], splits["val"], splits["test"]
    seen = set(scaf[tr]) | set(scaf[va])
    # 泄漏检查：scaffold split 下测试集骨架必须完全没在 train/val 出现过，期望值为 0
    seen_ratio = float(np.mean([s in seen for s in scaf[te]])) if len(te) else 0.0
    group_sizes = {}
    for name, idx in (("train", tr), ("val", va), ("test", te)):
        group_sizes[name] = int(len(set(scaf[idx])))
    return {
        "n_train": int(len(tr)),
        "n_val": int(len(va)),
        "n_test": int(len(te)),
        "n_scaffold_groups": int(len(set(scaffolds))),
        "n_scaffold_groups_per_split": group_sizes,
        "test_frac_scaffold_seen_in_train": round(seen_ratio, 4),
        "y_mean": {name: round(float(np.mean(y[idx])), 4) for name, idx in
                  (("train", tr), ("val", va), ("test", te))},
        "y_std": {name: round(float(np.std(y[idx])), 4) for name, idx in
                 (("train", tr), ("val", va), ("test", te))},
    }


# --------------------------------------------------------------------------- #
# 4. F1 预处理（只在 train 上 fit）
# --------------------------------------------------------------------------- #
def _nonfinite_to_nan(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    return np.where(np.isfinite(X), X, np.nan)


class F1Preprocessor:
    """design_plan 第 3.3 节的 F1 特征链；所有统计量只在 fit 的样本上计算。"""

    def __init__(self, corr_threshold: float = CORR_THRESHOLD,
                 protected: tuple[str, ...] = (LOGD_NAME,), skew: bool = False):
        self.corr_threshold = corr_threshold
        self.protected = protected
        self.skew = skew

    # ---- ① 填补 ----
    def _impute(self, X: np.ndarray) -> np.ndarray:
        return self.imputer_.transform(_nonfinite_to_nan(X))

    def _apply_skew(self, X: np.ndarray) -> np.ndarray:
        if not self.skew:
            return X
        X = X.copy()
        X[:, self.skew_mask_] = np.log1p(X[:, self.skew_mask_] + self.skew_shift_[self.skew_mask_])
        return X

    def fit(self, X: np.ndarray, y: np.ndarray, names: list[str]) -> "F1Preprocessor":
        X = _nonfinite_to_nan(X)
        y = np.asarray(y, dtype=float)
        self.n_features_in_ = X.shape[1]

        self.imputer_ = SimpleImputer(strategy="median", keep_empty_features=True)
        Xi = self.imputer_.fit_transform(X)
        Xi = np.nan_to_num(Xi, nan=0.0, posinf=0.0, neginf=0.0)

        if self.skew:
            from scipy.stats import skew as _skew
            with np.errstate(invalid="ignore", divide="ignore"):
                col_skew = np.nan_to_num(_skew(Xi, axis=0), nan=0.0)
            self.skew_mask_ = np.abs(col_skew) > 1.0
            # 先平移保证非负，再 log1p
            self.skew_shift_ = np.maximum(0.0, -np.min(Xi, axis=0)) + 1e-9
            Xi = self._apply_skew(Xi)
        else:
            self.skew_mask_ = np.zeros(Xi.shape[1], dtype=bool)
            self.skew_shift_ = np.zeros(Xi.shape[1])

        # ---- ② 去常量列 ----
        self.var_ = VarianceThreshold(0.0).fit(Xi)
        var_mask = self.var_.get_support()
        Xv = self.var_.transform(Xi)
        names_v = np.asarray(names, dtype=object)[var_mask]

        # ---- ④ 标准化 ----
        self.scaler_ = StandardScaler().fit(Xv)
        Xs = self.scaler_.transform(Xv)

        # ---- ⑤ 共线性剪枝 ----
        keep = self._corr_prune(Xs, y, names_v)
        self.support_ = keep
        self.feature_names_ = names_v[keep].tolist()
        self.dropped_by_var_ = int((~var_mask).sum())
        self.dropped_by_corr_ = int((~keep).sum())
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        Xi = self._impute(X)
        Xi = np.nan_to_num(Xi, nan=0.0, posinf=0.0, neginf=0.0)
        Xi = self._apply_skew(Xi)
        Xs = self.scaler_.transform(self.var_.transform(Xi))
        return Xs[:, self.support_]

    def _corr_prune(self, X: np.ndarray, y: np.ndarray, names: np.ndarray) -> np.ndarray:
        p = X.shape[1]
        keep = np.ones(p, dtype=bool)
        if p < 2:
            return keep

        with np.errstate(invalid="ignore", divide="ignore"):
            corr = np.nan_to_num(np.abs(np.corrcoef(X, rowvar=False)), nan=0.0)
            ry = np.array([abs(np.corrcoef(X[:, j], y)[0, 1]) if np.std(X[:, j]) > 0 else 0.0
                           for j in range(p)])
        ry = np.nan_to_num(ry, nan=0.0)

        protected = np.array([n in self.protected for n in names], dtype=bool)
        iu = np.triu_indices(p, k=1)
        pair_corr = corr[iu]
        for k in np.argsort(-pair_corr):
            if pair_corr[k] <= self.corr_threshold:
                break
            i, j = int(iu[0][k]), int(iu[1][k])
            if not (keep[i] and keep[j]):
                continue
            if protected[i] and not protected[j]:
                keep[j] = False
            elif protected[j] and not protected[i]:
                keep[i] = False
            elif ry[i] >= ry[j]:
                keep[j] = False
            else:
                keep[i] = False
        return keep

    def state(self) -> dict:
        return {
            "n_features_in": int(self.n_features_in_),
            "n_after_f1": int(self.support_.sum()),
            "dropped_constant": self.dropped_by_var_,
            "dropped_collinear": self.dropped_by_corr_,
            "corr_threshold": self.corr_threshold,
            "skew_correction": bool(self.skew),
            "protected_features": list(self.protected),
        }


# --------------------------------------------------------------------------- #
# 5. 指标
# --------------------------------------------------------------------------- #
def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) > 2:
        res = spearmanr(y_true, y_pred)
        rho = getattr(res, "statistic", None)
        if rho is None:
            rho = res.correlation
    else:
        rho = float("nan")
    return {
        "n": int(len(y_true)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else float("nan"),
        "spearman": float(rho) if rho is not None else float("nan"),
    }


def _fmt(metrics: dict) -> str:
    return (f"RMSE={metrics['rmse']:.4f}  MAE={metrics['mae']:.4f}  "
            f"R2={metrics['r2']:.4f}  rho={metrics['spearman']:.4f}  n={metrics['n']}")


# --------------------------------------------------------------------------- #
# 6. 训练
# --------------------------------------------------------------------------- #
def run_train(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log("== 1/5 load data ==")
    df = load_dataset(Path(args.data), require_target=True)
    log(f"  rows={len(df)}  cols={list(df.columns)}")

    smiles = df[SMILES_COL].tolist()
    y = df[TARGET_COL].to_numpy(dtype=float)

    log("== 2/5 features (RDKit descriptors + MolLogD) ==")
    X, bad, names = build_feature_matrix(
        smiles, cache_path=out_dir / "features_cache.npz",
        n_jobs=args.n_jobs, use_cache=not args.no_cache,
    )
    if bad.any():
        raise SystemExit(f"[error] {int(bad.sum())} SMILES in the training set cannot be parsed by RDKit; fix them in the cleaning stage")
    log(f"  feature matrix: {X.shape[0]} x {X.shape[1]} (including {LOGD_NAME})")

    log("== 3/5 scaffold split (8:1:1) ==")
    scaffolds = compute_scaffolds(smiles)
    splits = scaffold_split(scaffolds, SPLIT_RATIOS, seed=args.seed)
    split_info = split_report(splits, scaffolds, y)
    tr, va, te = splits["train"], splits["val"], splits["test"]
    log(f"  train={len(tr)}  val={len(va)}  test={len(te)}  "
        f"test scaffolds already seen in train/val={split_info['test_frac_scaffold_seen_in_train']:.1%}")

    log("== 4/5 F1 preprocessing + Ridge (alpha chosen by GroupKFold inside train) ==")
    pre = F1Preprocessor(corr_threshold=args.corr, skew=args.skew).fit(X[tr], y[tr], names)
    if LOGD_NAME not in pre.feature_names_:
        raise SystemExit(f"[error] {LOGD_NAME} did not survive preprocessing; check the protected-feature logic")
    Z_tr, Z_va, Z_te = pre.transform(X[tr]), pre.transform(X[va]), pre.transform(X[te])
    pre_info = pre.state()
    log(f"  features {pre_info['n_features_in']} -> {pre_info['n_after_f1']} "
        f"(constant -{pre_info['dropped_constant']}, collinear -{pre_info['dropped_collinear']})")

    scaf_tr = np.asarray([scaffolds[i] for i in tr], dtype=object)
    n_groups = len(set(scaf_tr))
    cv = GroupKFold(n_splits=max(2, min(args.cv, n_groups)))
    search = GridSearchCV(
        estimator=Ridge(),
        param_grid={"alpha": ALPHAS},
        cv=cv,
        scoring="neg_root_mean_squared_error",
        n_jobs=args.n_jobs,
        refit=True,
    )
    search.fit(Z_tr, y[tr], groups=scaf_tr)
    alpha = float(search.best_params_["alpha"])
    cv_rmse = float(-search.best_score_)
    cv_table = {f"{a:.4g}": round(float(-s), 4) for a, s in
                zip(search.cv_results_["param_alpha"], search.cv_results_["mean_test_score"])}
    log(f"  best alpha={alpha:.4g}  train-CV RMSE={cv_rmse:.4f}  (groups={n_groups})")

    log("== 5/5 evaluation ==")
    model_train = search.best_estimator_                     # 只用 train 拟合
    m_val = regression_metrics(y[va], model_train.predict(Z_va))
    m_test_holdout = regression_metrics(y[te], model_train.predict(Z_te))

    # 最终模型：train + val 重训（val 已经用过一次做诊断，这里按常规把它并入训练）
    final = Ridge(alpha=alpha).fit(np.vstack([Z_tr, Z_va]), np.concatenate([y[tr], y[va]]))
    test_predictions = final.predict(Z_te)
    m_test_final = regression_metrics(y[te], test_predictions)

    log(f"  [val]        {_fmt(m_val)}")
    log(f"  [test, train-only]  {_fmt(m_test_holdout)}")
    log(f"  [test, train+val]   {_fmt(m_test_final)}   <- final model")

    test_predictions_df = df.iloc[te][[ID_COL, SMILES_COL, TARGET_COL]].copy()
    test_predictions_df["pred_logD"] = test_predictions
    test_predictions_df["residual"] = test_predictions_df[TARGET_COL] - test_predictions_df["pred_logD"]
    test_predictions_path = out_dir / "test_predictions.csv"
    test_predictions_df.to_csv(test_predictions_path, index=False, encoding="utf-8-sig")

    bundle = {
        "preprocessor": pre,
        "model": final,
        "alpha": alpha,
        "feature_names": pre.feature_names_,
        "descriptor_names": names,
        "logd_feature": LOGD_NAME,
        "meta": {
            "data": str(args.data),
            "seed": int(args.seed),
            "split_ratios": list(SPLIT_RATIOS),
            "rdkit": rdBase.rdkitVersion,
            "trained_on": "train+val",
            "n_features": int(pre_info["n_after_f1"]),
        },
    }
    joblib.dump(bundle, out_dir / "model.joblib")

    result = {
        "config": {
            "data": str(args.data),
            "seed": int(args.seed),
            "split_ratios": list(SPLIT_RATIOS),
            "corr_threshold": float(args.corr),
            "skew_correction": bool(args.skew),
            "cv_folds": int(cv.n_splits),
            "alpha_grid": [float(a) for a in ALPHAS],
            "rdkit_version": rdBase.rdkitVersion,
        },
        "split": split_info,
        "features": pre_info,
        "model": {
            "type": "Ridge",
            "alpha": alpha,
            "train_cv_rmse_by_alpha": cv_table,
            "train_cv_rmse": cv_rmse,
        },
        "metrics": {
            "val": m_val,
            "test_train_only": m_test_holdout,
            "test_final": m_test_final,
        },
    }
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)

    with open(out_dir / "splits.json", "w", encoding="utf-8") as fh:
        json.dump({
            "seed": int(args.seed),
            "ratios": list(SPLIT_RATIOS),
            "train": tr.tolist(),
            "val": va.tolist(),
            "test": te.tolist(),
        }, fh, indent=2)

    log(f"  saved: {out_dir / 'model.joblib'}, metrics.json, splits.json, {test_predictions_path}")
    return 0


# --------------------------------------------------------------------------- #
# 7. 推理 / 检验接口
# --------------------------------------------------------------------------- #
def run_predict(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir)
    model_path = out_dir / "model.joblib"
    if not model_path.exists():
        raise SystemExit(f"[error] model not found at {model_path}; run `python test` first")

    bundle = joblib.load(model_path)
    input_path = Path(args.predict)
    df = load_dataset(input_path, require_target=False)
    log(f"  predict rows={len(df)} from {input_path}")

    smiles = df[SMILES_COL].tolist()
    X, bad, names = build_feature_matrix(smiles, cache_path=None, n_jobs=args.n_jobs,
                                         use_cache=False)
    if names != bundle["descriptor_names"]:
        raise SystemExit("[error] descriptor columns differ from training time (RDKit version changed?); retrain the model")

    out_dir.mkdir(parents=True, exist_ok=True)
    if bad.any():
        invalid = df.loc[bad].copy()
        invalid.to_csv(out_dir / "invalid_rows.csv", index=False, encoding="utf-8-sig")
        log(f"  warning: {int(bad.sum())} SMILES cannot be parsed; written to invalid_rows.csv")

    valid = ~bad
    if valid.sum() == 0:
        raise SystemExit("[error] no parsable molecules in input, nothing to predict")

    Z = bundle["preprocessor"].transform(X[valid])
    pred = bundle["model"].predict(Z)

    out_cols = [c for c in (ID_COL, SMILES_COL, TARGET_COL) if c in df.columns]
    result = df.loc[valid, out_cols].copy()
    result["pred"] = pred
    has_target = TARGET_COL in df.columns and df.loc[valid, TARGET_COL].notna().any()
    if has_target:
        result["residual"] = result[TARGET_COL] - result["pred"]
    result.to_csv(out_dir / "predictions.csv", index=False, encoding="utf-8-sig")
    log(f"  predictions -> {out_dir / 'predictions.csv'}")

    if has_target:
        y_all = df.loc[valid, TARGET_COL].to_numpy(dtype=float)
        mask = ~np.isnan(y_all)
        metrics = regression_metrics(y_all[mask], pred[mask])
        resid = y_all[mask] - pred[mask]
        report = {
            "input": str(input_path),
            "n_rows": int(len(df)),
            "n_invalid_smiles": int(bad.sum()),
            "n_scored": int(metrics["n"]),
            "metrics": metrics,
            "residual": {
                "mean": float(np.mean(resid)),
                "std": float(np.std(resid)),
            },
            "alpha": bundle["alpha"],
            "n_features": bundle["meta"]["n_features"],
        }
        with open(out_dir / "predict_metrics.json", "w", encoding="utf-8") as fh:
            json.dump(report, fh, ensure_ascii=False, indent=2)
        log(f"  [predict] {_fmt(metrics)}")
        log(f"  report -> {out_dir / 'predict_metrics.json'}")
    else:
        log("  input has no usable exp column -> predictions only, no metrics")
    return 0


# --------------------------------------------------------------------------- #
# 8. CLI
# --------------------------------------------------------------------------- #
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="logD regression: RDKit descriptors (F1) + MolLogD + Ridge",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data", default="Lipophilicity_clean.csv", help="training CSV")
    parser.add_argument("--out-dir", default="outputs", help="output directory")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="seed for the scaffold split")
    parser.add_argument("--corr", type=float, default=CORR_THRESHOLD, help="collinearity pruning threshold")
    parser.add_argument("--cv", type=int, default=5, help="GroupKFold folds inside train")
    parser.add_argument("--skew", action="store_true",
                        help="enable skew correction (optional branch in design_plan)")
    parser.add_argument("--n-jobs", type=int, default=-1, help="worker processes, -1 uses all cores")
    parser.add_argument("--no-cache", action="store_true", help="ignore the feature cache and recompute")
    parser.add_argument("--predict", default=None, metavar="CSV",
                        help="inference mode: predict a CSV with the saved model and report metrics")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    log(f"python {sys.version.split()[0]}  rdkit {rdBase.rdkitVersion}")
    if args.predict:
        return run_predict(args)
    return run_train(args)


if __name__ == "__main__":
    raise SystemExit(main())

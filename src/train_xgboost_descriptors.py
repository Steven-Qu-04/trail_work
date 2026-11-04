"""用 Morgan+理化描述符和 Scaffold 交叉验证优化 XGBoost。"""

import hashlib
import json
import platform
import statistics
import time

import joblib
import numpy as np
import pandas as pd
import rdkit
import sklearn
import xgboost
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor

from features import (
    DEFAULT_DESCRIPTOR_NAMES,
    DEFAULT_FP_SIZE,
    DEFAULT_RADIUS,
    smiles_list_to_feature_matrix,
)
from train import (
    MODEL_DIR,
    PROJECT_DIR,
    XGBOOST_REPORT_DIR,
    TEST_PATH,
    TRAIN_PATH,
    VALID_PATH,
    calculate_metrics,
)


MODEL_PATH = MODEL_DIR / "xgboost_descriptors_model.joblib"
REPORT_PATH = XGBOOST_REPORT_DIR / "xgboost_descriptors_metrics.json"
CV_CSV_PATH = XGBOOST_REPORT_DIR / "xgboost_descriptors_cv_results.csv"
CV_JSON_PATH = XGBOOST_REPORT_DIR / "xgboost_descriptors_cv_results.json"
TEST_PREDICTIONS_PATH = (
    XGBOOST_REPORT_DIR / "xgboost_descriptors_scaffold_test_predictions.csv"
)

FEATURE_CONFIG = {
    "type": "Morgan+RDKitDescriptors",
    "radius": DEFAULT_RADIUS,
    "fp_size": DEFAULT_FP_SIZE,
    "descriptor_names": list(DEFAULT_DESCRIPTOR_NAMES),
}

BASE_PARAMS = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "tree_method": "hist",
    "n_estimators": 1800,
    "early_stopping_rounds": 50,
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": 0,
}

CANDIDATES = [
    {"max_depth": 5, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.6, "min_child_weight": 1, "reg_lambda": 5.0, "reg_alpha": 0.0, "gamma": 0.0},
    {"max_depth": 5, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.6, "min_child_weight": 1, "reg_lambda": 5.0, "reg_alpha": 0.1, "gamma": 0.0},
    {"max_depth": 5, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.6, "min_child_weight": 1, "reg_lambda": 5.0, "reg_alpha": 0.5, "gamma": 0.0},
    {"max_depth": 5, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.6, "min_child_weight": 1, "reg_lambda": 10.0, "reg_alpha": 0.0, "gamma": 0.0},
    {"max_depth": 5, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.6, "min_child_weight": 3, "reg_lambda": 5.0, "reg_alpha": 0.0, "gamma": 0.0},
    {"max_depth": 4, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.6, "min_child_weight": 1, "reg_lambda": 5.0, "reg_alpha": 0.0, "gamma": 0.0},
    {"max_depth": 6, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.6, "min_child_weight": 1, "reg_lambda": 5.0, "reg_alpha": 0.0, "gamma": 0.0},
    {"max_depth": 5, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 1, "reg_lambda": 5.0, "reg_alpha": 0.0, "gamma": 0.0},
    {"max_depth": 5, "learning_rate": 0.03, "subsample": 0.8, "colsample_bytree": 0.6, "min_child_weight": 1, "reg_lambda": 5.0, "reg_alpha": 0.0, "gamma": 0.0},
    {"max_depth": 5, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.6, "min_child_weight": 1, "reg_lambda": 5.0, "reg_alpha": 0.0, "gamma": 0.05},
]


def load_combined_split(path):
    """读取固定划分，并生成 Morgan+描述符矩阵。"""
    if not path.exists():
        raise FileNotFoundError(f"找不到数据文件：{path}")

    df = pd.read_csv(path)
    missing_columns = {"smiles", "exp"} - set(df.columns)
    if missing_columns:
        raise ValueError(f"{path.name} 缺少必要列：{sorted(missing_columns)}")

    labels = pd.to_numeric(df["exp"], errors="coerce")
    if labels.isna().any():
        raise ValueError(f"{path.name} 包含无效 exp")

    features, invalid_positions = smiles_list_to_feature_matrix(
        df["smiles"].tolist(),
        FEATURE_CONFIG,
    )
    if invalid_positions:
        bad_rows = [position + 2 for position in invalid_positions]
        raise ValueError(f"{path.name} 包含无效 SMILES，CSV行号：{bad_rows}")

    return df, features, labels.to_numpy(dtype=float)


def scaffold_from_smiles(smiles: str) -> str:
    """使用与固定数据划分一致的规则生成 scaffold 分组键。"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"无效 SMILES：{smiles}")
    if mol.GetRingInfo().NumRings() == 0:
        digest = hashlib.sha256(smiles.strip().encode("utf-8")).hexdigest()
        return "ACYCLIC_" + digest

    scaffold = MurckoScaffold.MurckoScaffoldSmiles(
        mol=mol,
        includeChirality=False,
    )
    if not scaffold:
        raise RuntimeError(f"含环分子未能生成 scaffold：{smiles}")
    return scaffold


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    XGBOOST_REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("生成训练集 Morgan+RDKit描述符特征...", flush=True)
    train_df, x_train, y_train = load_combined_split(TRAIN_PATH)
    scaffold_groups = np.asarray(
        [scaffold_from_smiles(smiles) for smiles in train_df["smiles"]],
        dtype=object,
    )
    print(
        f"训练集 {len(train_df)} 条，特征维数 {x_train.shape[1]}，"
        f"scaffold组数 {len(set(scaffold_groups))}"
    )

    splitter = GroupKFold(n_splits=5, shuffle=True, random_state=42)
    folds = list(splitter.split(x_train, y_train, groups=scaffold_groups))
    cv_results = []

    for candidate_number, candidate in enumerate(CANDIDATES, start=1):
        print(
            f"[{candidate_number}/{len(CANDIDATES)}] 5折Scaffold CV：{candidate}",
            flush=True,
        )
        fold_metrics = []
        fold_tree_counts = []
        start_time = time.perf_counter()

        for fold_number, (fit_indices, fold_valid_indices) in enumerate(
            folds,
            start=1,
        ):
            fit_groups = set(scaffold_groups[fit_indices])
            fold_valid_groups = set(scaffold_groups[fold_valid_indices])
            if not fit_groups.isdisjoint(fold_valid_groups):
                raise RuntimeError(f"第 {fold_number} 折存在 scaffold 交叉")

            model = XGBRegressor(**{**BASE_PARAMS, **candidate})
            model.fit(
                x_train[fit_indices],
                y_train[fit_indices],
                eval_set=[
                    (x_train[fold_valid_indices], y_train[fold_valid_indices])
                ],
                verbose=False,
            )
            predictions = model.predict(x_train[fold_valid_indices])
            metrics = calculate_metrics(
                y_train[fold_valid_indices],
                predictions,
            )
            fold_metrics.append(metrics)
            fold_tree_counts.append(int(model.best_iteration) + 1)

        elapsed_seconds = time.perf_counter() - start_time
        mean_rmse = float(np.mean([item["RMSE"] for item in fold_metrics]))
        mean_mae = float(np.mean([item["MAE"] for item in fold_metrics]))
        mean_r2 = float(np.mean([item["R2"] for item in fold_metrics]))
        std_rmse = float(np.std([item["RMSE"] for item in fold_metrics]))
        median_trees = int(round(statistics.median(fold_tree_counts)))

        cv_results.append(
            {
                "candidate": candidate_number,
                **candidate,
                "cv_mean_RMSE": mean_rmse,
                "cv_std_RMSE": std_rmse,
                "cv_mean_MAE": mean_mae,
                "cv_mean_R2": mean_r2,
                "median_n_estimators": median_trees,
                "fold_RMSE": [item["RMSE"] for item in fold_metrics],
                "fold_n_estimators": fold_tree_counts,
                "elapsed_seconds": elapsed_seconds,
            }
        )
        print(
            f"    CV RMSE={mean_rmse:.6f}±{std_rmse:.6f}, "
            f"MAE={mean_mae:.6f}, R²={mean_r2:.6f}, "
            f"中位树数={median_trees}, 耗时={elapsed_seconds:.1f}s",
            flush=True,
        )

    cv_results.sort(key=lambda item: item["cv_mean_RMSE"])
    best_result = cv_results[0]
    parameter_names = (
        "max_depth",
        "learning_rate",
        "subsample",
        "colsample_bytree",
        "min_child_weight",
        "reg_lambda",
        "reg_alpha",
        "gamma",
    )
    best_params = {
        "objective": BASE_PARAMS["objective"],
        "eval_metric": BASE_PARAMS["eval_metric"],
        "tree_method": BASE_PARAMS["tree_method"],
        "n_estimators": best_result["median_n_estimators"],
        "random_state": BASE_PARAMS["random_state"],
        "n_jobs": BASE_PARAMS["n_jobs"],
        "verbosity": BASE_PARAMS["verbosity"],
        **{name: best_result[name] for name in parameter_names},
    }

    csv_rows = [
        {key: value for key, value in result.items() if not isinstance(value, list)}
        for result in cv_results
    ]
    pd.DataFrame(csv_rows).to_csv(CV_CSV_PATH, index=False)
    with CV_JSON_PATH.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "selection_metric": "5-fold scaffold CV mean RMSE",
                "cv_source": TRAIN_PATH.name,
                "validation_file_used_for_selection": False,
                "internal_test_used_for_selection": False,
                "feature_config": FEATURE_CONFIG,
                "results": cv_results,
                "best_params": best_params,
            },
            file,
            indent=2,
            ensure_ascii=False,
        )

    print("\nCV最优参数：", best_params)
    print("在完整训练集拟合并评估独立验证集...", flush=True)
    valid_df, x_valid, y_valid = load_combined_split(VALID_PATH)
    validation_model = XGBRegressor(**best_params)
    validation_model.fit(x_train, y_train, verbose=False)
    train_metrics = calculate_metrics(y_train, validation_model.predict(x_train))
    valid_metrics = calculate_metrics(y_valid, validation_model.predict(x_valid))

    print("使用训练集和验证集拟合最终模型...", flush=True)
    x_final = np.concatenate([x_train, x_valid], axis=0)
    y_final = np.concatenate([y_train, y_valid], axis=0)
    final_model = XGBRegressor(**best_params)
    final_model.fit(x_final, y_final, verbose=False)

    print("最终模型已确定，开始读取内部测试集...", flush=True)
    test_df, x_test, y_test = load_combined_split(TEST_PATH)
    test_pred = final_model.predict(x_test)
    test_metrics = calculate_metrics(y_test, test_pred)

    model_bundle = {
        "model": final_model,
        "model_type": "XGBRegressor",
        "feature_config": FEATURE_CONFIG,
        "model_params": best_params,
        "parameter_selection": "5-fold scaffold CV on train.csv",
        "target_column": "exp",
        "training_files": [TRAIN_PATH.name, VALID_PATH.name],
        "internal_test_file": TEST_PATH.name,
    }
    joblib.dump(model_bundle, MODEL_PATH, compress=3)

    test_results = test_df.copy()
    test_results["predicted_exp"] = test_pred
    test_results["residual"] = y_test - test_pred
    test_results.to_csv(TEST_PREDICTIONS_PATH, index=False)

    report = {
        "data": {
            "train_rows": len(train_df),
            "validation_rows": len(valid_df),
            "internal_test_rows": len(test_df),
            "scaffold_cv_folds": 5,
            "scaffold_cv_groups": len(set(scaffold_groups)),
        },
        "model_type": "XGBRegressor",
        "feature_config": FEATURE_CONFIG,
        "model_params": best_params,
        "parameter_selection": {
            "metric": "5-fold scaffold CV mean RMSE",
            "cv_report": CV_JSON_PATH.name,
            "validation_file_used_for_selection": False,
            "internal_test_used_for_selection": False,
        },
        "scaffold_cv": {
            "mean_RMSE": best_result["cv_mean_RMSE"],
            "std_RMSE": best_result["cv_std_RMSE"],
            "mean_MAE": best_result["cv_mean_MAE"],
            "mean_R2": best_result["cv_mean_R2"],
        },
        "xgboost_descriptors": {
            "train_before_refit": train_metrics,
            "validation_before_refit": valid_metrics,
            "internal_test_after_train_validation_refit": test_metrics,
        },
        "environment": {
            "python": platform.python_version(),
            "rdkit": rdkit.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
            "xgboost": xgboost.__version__,
        },
    }
    with REPORT_PATH.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    print("\n===== XGBoost + 描述符评估结果 =====")
    print("5折Scaffold CV：", report["scaffold_cv"])
    print("训练集（重新拟合前）：", train_metrics)
    print("验证集（重新拟合前）：", valid_metrics)
    print("内部测试集（最终模型）：", test_metrics)
    print(f"模型已保存：{MODEL_PATH.relative_to(PROJECT_DIR)}")
    print(f"报告已保存：{REPORT_PATH.relative_to(PROJECT_DIR)}")


if __name__ == "__main__":
    main()

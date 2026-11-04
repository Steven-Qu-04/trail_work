"""搜索 XGBoost 参数，训练最终模型并评估固定内部测试集。"""

import json
import platform
import time

import joblib
import numpy as np
import pandas as pd
import rdkit
import sklearn
import xgboost
from xgboost import XGBRegressor

from train import (
    FEATURE_CONFIG,
    MODEL_DIR,
    PROJECT_DIR,
    REPORT_DIR,
    TEST_PATH,
    TRAIN_PATH,
    VALID_PATH,
    calculate_metrics,
    load_split,
)


MODEL_PATH = MODEL_DIR / "xgboost_model.joblib"
REPORT_PATH = REPORT_DIR / "xgboost_metrics.json"
TUNING_CSV_PATH = REPORT_DIR / "xgboost_tuning_results.csv"
TUNING_JSON_PATH = REPORT_DIR / "xgboost_tuning_results.json"
TEST_PREDICTIONS_PATH = REPORT_DIR / "xgboost_scaffold_test_predictions.csv"

BASE_PARAMS = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "tree_method": "hist",
    "n_estimators": 2000,
    "early_stopping_rounds": 60,
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": 0,
}

CANDIDATES = [
    {"max_depth": 3, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.3, "min_child_weight": 1, "reg_lambda": 1.0},
    {"max_depth": 3, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.6, "min_child_weight": 1, "reg_lambda": 1.0},
    {"max_depth": 3, "learning_rate": 0.05, "subsample": 1.0, "colsample_bytree": 0.6, "min_child_weight": 1, "reg_lambda": 1.0},
    {"max_depth": 3, "learning_rate": 0.1, "subsample": 0.8, "colsample_bytree": 0.6, "min_child_weight": 1, "reg_lambda": 1.0},
    {"max_depth": 5, "learning_rate": 0.03, "subsample": 0.8, "colsample_bytree": 0.6, "min_child_weight": 1, "reg_lambda": 1.0},
    {"max_depth": 5, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.3, "min_child_weight": 1, "reg_lambda": 1.0},
    {"max_depth": 5, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.6, "min_child_weight": 1, "reg_lambda": 1.0},
    {"max_depth": 5, "learning_rate": 0.1, "subsample": 0.8, "colsample_bytree": 0.6, "min_child_weight": 1, "reg_lambda": 1.0},
    {"max_depth": 7, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.3, "min_child_weight": 1, "reg_lambda": 1.0},
    {"max_depth": 7, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.6, "min_child_weight": 1, "reg_lambda": 1.0},
    {"max_depth": 5, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.6, "min_child_weight": 5, "reg_lambda": 1.0},
    {"max_depth": 5, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.6, "min_child_weight": 1, "reg_lambda": 5.0},
]


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("读取训练集和验证集并生成 Morgan 指纹...", flush=True)
    train_df, x_train, y_train = load_split(TRAIN_PATH)
    valid_df, x_valid, y_valid = load_split(VALID_PATH)
    print(f"训练集 {len(train_df)} 条，验证集 {len(valid_df)} 条")

    tuning_results = []
    for candidate_number, candidate in enumerate(CANDIDATES, start=1):
        params = {**BASE_PARAMS, **candidate}
        print(
            f"[{candidate_number}/{len(CANDIDATES)}] XGBoost：{candidate}",
            flush=True,
        )
        start_time = time.perf_counter()
        model = XGBRegressor(**params)
        model.fit(
            x_train,
            y_train,
            eval_set=[(x_valid, y_valid)],
            verbose=False,
        )
        train_metrics = calculate_metrics(y_train, model.predict(x_train))
        valid_metrics = calculate_metrics(y_valid, model.predict(x_valid))
        elapsed_seconds = time.perf_counter() - start_time
        best_iteration = int(model.best_iteration)

        tuning_results.append(
            {
                "candidate": candidate_number,
                **candidate,
                "best_n_estimators": best_iteration + 1,
                "train_RMSE": train_metrics["RMSE"],
                "train_MAE": train_metrics["MAE"],
                "train_R2": train_metrics["R2"],
                "valid_RMSE": valid_metrics["RMSE"],
                "valid_MAE": valid_metrics["MAE"],
                "valid_R2": valid_metrics["R2"],
                "elapsed_seconds": elapsed_seconds,
            }
        )
        print(
            f"    最佳树数={best_iteration + 1}, "
            f"验证 RMSE={valid_metrics['RMSE']:.6f}, "
            f"MAE={valid_metrics['MAE']:.6f}, R²={valid_metrics['R2']:.6f}, "
            f"耗时={elapsed_seconds:.1f}s",
            flush=True,
        )

    tuning_results.sort(key=lambda item: item["valid_RMSE"])
    best_result = tuning_results[0]
    candidate_names = (
        "max_depth",
        "learning_rate",
        "subsample",
        "colsample_bytree",
        "min_child_weight",
        "reg_lambda",
    )
    best_params = {
        "objective": BASE_PARAMS["objective"],
        "eval_metric": BASE_PARAMS["eval_metric"],
        "tree_method": BASE_PARAMS["tree_method"],
        "n_estimators": best_result["best_n_estimators"],
        "random_state": BASE_PARAMS["random_state"],
        "n_jobs": BASE_PARAMS["n_jobs"],
        "verbosity": BASE_PARAMS["verbosity"],
        **{name: best_result[name] for name in candidate_names},
    }

    pd.DataFrame(tuning_results).to_csv(TUNING_CSV_PATH, index=False)
    with TUNING_JSON_PATH.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "selection_metric": "validation RMSE (lower is better)",
                "internal_test_used_for_selection": False,
                "early_stopping_rounds": BASE_PARAMS["early_stopping_rounds"],
                "results": tuning_results,
                "best_params": best_params,
            },
            file,
            indent=2,
            ensure_ascii=False,
        )

    print("\n验证集最优参数：", best_params)
    print("使用训练集和验证集拟合最终 XGBoost 模型...", flush=True)
    x_final = np.concatenate([x_train, x_valid], axis=0)
    y_final = np.concatenate([y_train, y_valid], axis=0)
    final_model = XGBRegressor(**best_params)
    final_model.fit(x_final, y_final, verbose=False)

    print("最终模型已确定，开始读取内部测试集...", flush=True)
    test_df, x_test, y_test = load_split(TEST_PATH)
    test_pred = final_model.predict(x_test)
    test_metrics = calculate_metrics(y_test, test_pred)

    model_bundle = {
        "model": final_model,
        "model_type": "XGBRegressor",
        "feature_config": FEATURE_CONFIG,
        "model_params": best_params,
        "parameter_selection": "lowest validation RMSE with early stopping",
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
            "split_type": "Bemis-Murcko scaffold split",
        },
        "model_type": "XGBRegressor",
        "feature_config": FEATURE_CONFIG,
        "model_params": best_params,
        "parameter_selection": {
            "metric": "validation RMSE",
            "tuning_report": TUNING_JSON_PATH.name,
            "internal_test_used_for_selection": False,
        },
        "xgboost": {
            "train_before_refit": {
                "RMSE": best_result["train_RMSE"],
                "MAE": best_result["train_MAE"],
                "R2": best_result["train_R2"],
            },
            "validation_before_refit": {
                "RMSE": best_result["valid_RMSE"],
                "MAE": best_result["valid_MAE"],
                "R2": best_result["valid_R2"],
            },
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

    print("\n===== XGBoost 评估结果 =====")
    print("训练集（重新拟合前）：", report["xgboost"]["train_before_refit"])
    print("验证集（重新拟合前）：", report["xgboost"]["validation_before_refit"])
    print("内部测试集（最终模型）：", test_metrics)
    print(f"模型已保存：{MODEL_PATH.relative_to(PROJECT_DIR)}")
    print(f"报告已保存：{REPORT_PATH.relative_to(PROJECT_DIR)}")
    print(f"调参结果：{TUNING_CSV_PATH.relative_to(PROJECT_DIR)}")


if __name__ == "__main__":
    main()

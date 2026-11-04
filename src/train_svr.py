"""搜索 SVR 参数，训练最终模型并评估固定内部测试集。"""

import json
import platform
import time

import joblib
import numpy as np
import pandas as pd
import rdkit
import sklearn
from sklearn.svm import SVR

from train import (
    FEATURE_CONFIG,
    COMPONENT_MODEL_DIR,
    PROJECT_DIR,
    REPORT_DIR,
    TEST_PATH,
    TRAIN_PATH,
    VALID_PATH,
    calculate_metrics,
    load_split,
)


MODEL_PATH = COMPONENT_MODEL_DIR / "svr_model.joblib"
REPORT_PATH = REPORT_DIR / "svr_metrics.json"
TUNING_CSV_PATH = REPORT_DIR / "svr_tuning_results.csv"
TUNING_JSON_PATH = REPORT_DIR / "svr_tuning_results.json"
TEST_PREDICTIONS_PATH = REPORT_DIR / "svr_scaffold_test_predictions.csv"

BASE_PARAMS = {
    "cache_size": 1000,
    "tol": 0.001,
    "max_iter": -1,
}

# 先比较 RBF 核的惩罚强度和核宽度，再检查 epsilon 与线性核。
CANDIDATES = [
    {"kernel": "rbf", "C": 1.0, "gamma": "scale", "epsilon": 0.1},
    {"kernel": "rbf", "C": 3.0, "gamma": "scale", "epsilon": 0.1},
    {"kernel": "rbf", "C": 10.0, "gamma": "scale", "epsilon": 0.1},
    {"kernel": "rbf", "C": 30.0, "gamma": "scale", "epsilon": 0.1},
    {"kernel": "rbf", "C": 100.0, "gamma": "scale", "epsilon": 0.1},
    {"kernel": "rbf", "C": 10.0, "gamma": 0.001, "epsilon": 0.1},
    {"kernel": "rbf", "C": 10.0, "gamma": 0.005, "epsilon": 0.1},
    {"kernel": "rbf", "C": 10.0, "gamma": 0.01, "epsilon": 0.1},
    {"kernel": "rbf", "C": 10.0, "gamma": 0.02, "epsilon": 0.1},
    {"kernel": "rbf", "C": 10.0, "gamma": 0.05, "epsilon": 0.1},
    {"kernel": "rbf", "C": 30.0, "gamma": 0.005, "epsilon": 0.1},
    {"kernel": "rbf", "C": 30.0, "gamma": 0.01, "epsilon": 0.1},
    {"kernel": "rbf", "C": 30.0, "gamma": 0.02, "epsilon": 0.1},
    {"kernel": "rbf", "C": 100.0, "gamma": 0.005, "epsilon": 0.1},
    {"kernel": "rbf", "C": 10.0, "gamma": 0.01, "epsilon": 0.05},
    {"kernel": "rbf", "C": 10.0, "gamma": 0.01, "epsilon": 0.2},
    {"kernel": "linear", "C": 0.1, "gamma": "scale", "epsilon": 0.1},
    {"kernel": "linear", "C": 1.0, "gamma": "scale", "epsilon": 0.1},
]


def main() -> None:
    COMPONENT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("读取训练集和验证集并生成 Morgan 指纹...", flush=True)
    train_df, x_train, y_train = load_split(TRAIN_PATH)
    valid_df, x_valid, y_valid = load_split(VALID_PATH)
    print(f"训练集 {len(train_df)} 条，验证集 {len(valid_df)} 条")

    tuning_results = []
    for candidate_number, candidate in enumerate(CANDIDATES, start=1):
        params = {**BASE_PARAMS, **candidate}
        print(
            f"[{candidate_number}/{len(CANDIDATES)}] SVR：{candidate}",
            flush=True,
        )
        start_time = time.perf_counter()
        model = SVR(**params)
        model.fit(x_train, y_train)
        train_metrics = calculate_metrics(y_train, model.predict(x_train))
        valid_metrics = calculate_metrics(y_valid, model.predict(x_valid))
        elapsed_seconds = time.perf_counter() - start_time

        tuning_results.append(
            {
                "candidate": candidate_number,
                **candidate,
                "train_RMSE": train_metrics["RMSE"],
                "train_MAE": train_metrics["MAE"],
                "train_R2": train_metrics["R2"],
                "valid_RMSE": valid_metrics["RMSE"],
                "valid_MAE": valid_metrics["MAE"],
                "valid_R2": valid_metrics["R2"],
                "support_vectors": int(model.support_.size),
                "elapsed_seconds": elapsed_seconds,
            }
        )
        print(
            f"    验证 RMSE={valid_metrics['RMSE']:.6f}, "
            f"MAE={valid_metrics['MAE']:.6f}, R²={valid_metrics['R2']:.6f}, "
            f"支持向量={model.support_.size}, 耗时={elapsed_seconds:.1f}s",
            flush=True,
        )

    tuning_results.sort(key=lambda item: item["valid_RMSE"])
    best_result = tuning_results[0]
    best_params = {
        **BASE_PARAMS,
        **{
            name: best_result[name]
            for name in ("kernel", "C", "gamma", "epsilon")
        },
    }

    pd.DataFrame(tuning_results).to_csv(TUNING_CSV_PATH, index=False)
    with TUNING_JSON_PATH.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "selection_metric": "validation RMSE (lower is better)",
                "internal_test_used_for_selection": False,
                "feature_scaling": "none; Morgan bits are already binary",
                "results": tuning_results,
                "best_params": best_params,
            },
            file,
            indent=2,
            ensure_ascii=False,
        )

    print("\n验证集最优参数：", best_params)
    print("使用训练集和验证集拟合最终 SVR 模型...", flush=True)
    x_final = np.concatenate([x_train, x_valid], axis=0)
    y_final = np.concatenate([y_train, y_valid], axis=0)
    final_model = SVR(**best_params)
    final_model.fit(x_final, y_final)

    print("最终模型已确定，开始读取内部测试集...", flush=True)
    test_df, x_test, y_test = load_split(TEST_PATH)
    test_pred = final_model.predict(x_test)
    test_metrics = calculate_metrics(y_test, test_pred)

    model_bundle = {
        "model": final_model,
        "model_type": "SVR",
        "feature_config": FEATURE_CONFIG,
        "model_params": best_params,
        "feature_scaling": "none; Morgan bits are binary",
        "parameter_selection": "lowest validation RMSE",
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
        "model_type": "SVR",
        "feature_config": FEATURE_CONFIG,
        "model_params": best_params,
        "feature_scaling": "none; Morgan bits are binary",
        "parameter_selection": {
            "metric": "validation RMSE",
            "tuning_report": TUNING_JSON_PATH.name,
            "internal_test_used_for_selection": False,
        },
        "svr": {
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
            "final_support_vectors": int(final_model.support_.size),
        },
        "environment": {
            "python": platform.python_version(),
            "rdkit": rdkit.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
    }
    with REPORT_PATH.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    print("\n===== SVR 评估结果 =====")
    print("训练集（重新拟合前）：", report["svr"]["train_before_refit"])
    print("验证集（重新拟合前）：", report["svr"]["validation_before_refit"])
    print("内部测试集（最终模型）：", test_metrics)
    print(f"最终支持向量数：{final_model.support_.size}")
    print(f"模型已保存：{MODEL_PATH.relative_to(PROJECT_DIR)}")
    print(f"报告已保存：{REPORT_PATH.relative_to(PROJECT_DIR)}")
    print(f"调参结果：{TUNING_CSV_PATH.relative_to(PROJECT_DIR)}")


if __name__ == "__main__":
    main()

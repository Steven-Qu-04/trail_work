"""只使用训练集和验证集搜索 Random Forest 参数。"""

import json
from pathlib import Path
import time

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from train import (
    BASELINE_MODEL_DIR,
    BASELINE_REPORT_DIR,
    PROJECT_DIR,
    TRAIN_PATH,
    VALID_PATH,
    calculate_metrics,
    load_split,
)


REPORT_DIR = BASELINE_REPORT_DIR
MODEL_DIR = BASELINE_MODEL_DIR
RESULTS_CSV_PATH = REPORT_DIR / "rf_tuning_results.csv"
RESULTS_JSON_PATH = REPORT_DIR / "rf_tuning_results.json"
BEST_PARAMS_PATH = MODEL_DIR / "best_rf_params.json"

BASE_PARAMS = {
    "n_estimators": 300,
    "random_state": 42,
    "n_jobs": -1,
}

# 第一项是原模型，用于直接判断调参是否带来改进。
CANDIDATES = [
    {"max_features": "sqrt", "min_samples_leaf": 1, "max_depth": None},
    {"max_features": "sqrt", "min_samples_leaf": 2, "max_depth": None},
    {"max_features": 0.1, "min_samples_leaf": 1, "max_depth": None},
    {"max_features": 0.1, "min_samples_leaf": 2, "max_depth": None},
    {"max_features": 0.2, "min_samples_leaf": 1, "max_depth": None},
    {"max_features": 0.2, "min_samples_leaf": 2, "max_depth": None},
    {"max_features": 0.3, "min_samples_leaf": 1, "max_depth": None},
    {"max_features": 0.3, "min_samples_leaf": 2, "max_depth": None},
    {"max_features": 0.3, "min_samples_leaf": 4, "max_depth": None},
    {"max_features": 0.3, "min_samples_leaf": 2, "max_depth": 30},
    {"max_features": 0.5, "min_samples_leaf": 2, "max_depth": None},
]


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("读取训练集和验证集并生成 Morgan 指纹...", flush=True)
    _, x_train, y_train = load_split(TRAIN_PATH)
    _, x_valid, y_valid = load_split(VALID_PATH)

    results = []
    for candidate_number, candidate in enumerate(CANDIDATES, start=1):
        params = {**BASE_PARAMS, **candidate}
        print(
            f"[{candidate_number}/{len(CANDIDATES)}] 测试参数：{candidate}",
            flush=True,
        )
        start_time = time.perf_counter()
        model = RandomForestRegressor(**params)
        model.fit(x_train, y_train)
        train_metrics = calculate_metrics(y_train, model.predict(x_train))
        valid_metrics = calculate_metrics(y_valid, model.predict(x_valid))
        elapsed_seconds = time.perf_counter() - start_time

        result = {
            "candidate": candidate_number,
            **params,
            "train_RMSE": train_metrics["RMSE"],
            "train_MAE": train_metrics["MAE"],
            "train_R2": train_metrics["R2"],
            "valid_RMSE": valid_metrics["RMSE"],
            "valid_MAE": valid_metrics["MAE"],
            "valid_R2": valid_metrics["R2"],
            "elapsed_seconds": elapsed_seconds,
        }
        results.append(result)
        print(
            f"    验证 RMSE={valid_metrics['RMSE']:.6f}, "
            f"MAE={valid_metrics['MAE']:.6f}, "
            f"R²={valid_metrics['R2']:.6f}, "
            f"耗时={elapsed_seconds:.1f}s",
            flush=True,
        )

    results.sort(key=lambda item: item["valid_RMSE"])
    best_result = results[0]
    best_params = {
        key: best_result[key]
        for key in (
            "n_estimators",
            "max_features",
            "min_samples_leaf",
            "max_depth",
            "random_state",
            "n_jobs",
        )
    }

    pd.DataFrame(results).to_csv(RESULTS_CSV_PATH, index=False)
    with RESULTS_JSON_PATH.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "selection_metric": "validation RMSE (lower is better)",
                "train_file": TRAIN_PATH.name,
                "validation_file": VALID_PATH.name,
                "results": results,
                "best_params": best_params,
            },
            file,
            indent=2,
            ensure_ascii=False,
        )
    with BEST_PARAMS_PATH.open("w", encoding="utf-8") as file:
        json.dump(best_params, file, indent=2, ensure_ascii=False)

    print("\n===== 调参完成 =====")
    print("最优参数：", best_params)
    print(f"最优验证 RMSE：{best_result['valid_RMSE']:.6f}")
    print(f"结果表：{RESULTS_CSV_PATH.relative_to(PROJECT_DIR)}")
    print(f"参数文件：{BEST_PARAMS_PATH.relative_to(PROJECT_DIR)}")


if __name__ == "__main__":
    main()

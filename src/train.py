"""使用固定 Scaffold Split 训练并评估 Random Forest 基线模型。"""

import json
from pathlib import Path
import platform

import joblib
import numpy as np
import pandas as pd
import rdkit
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from features import DEFAULT_FP_SIZE, DEFAULT_RADIUS, smiles_list_to_matrix


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "processed"
MODEL_ROOT = PROJECT_DIR / "models"
REPORT_ROOT = PROJECT_DIR / "reports"
BASELINE_MODEL_DIR = MODEL_ROOT / "baselines"
COMPONENT_MODEL_DIR = MODEL_ROOT / "components"
PRODUCTION_MODEL_DIR = MODEL_ROOT / "production"
BASELINE_REPORT_DIR = REPORT_ROOT / "baselines"
XGBOOST_REPORT_DIR = REPORT_ROOT / "xgboost"
PRODUCTION_REPORT_DIR = REPORT_ROOT / "production"
TRAIN_PATH = DATA_DIR / "train.csv"
VALID_PATH = DATA_DIR / "valid.csv"
TEST_PATH = DATA_DIR / "scaffold_test.csv"
# Backward-compatible aliases used by the baseline training scripts.
MODEL_DIR = BASELINE_MODEL_DIR
REPORT_DIR = BASELINE_REPORT_DIR
MODEL_PATH = MODEL_DIR / "logd_model.joblib"
REPORT_PATH = REPORT_DIR / "metrics.json"
TEST_PREDICTIONS_PATH = REPORT_DIR / "scaffold_test_predictions.csv"

FEATURE_CONFIG = {
    "type": "Morgan",
    "radius": DEFAULT_RADIUS,
    "fp_size": DEFAULT_FP_SIZE,
}
MODEL_PARAMS = {
    "n_estimators": 300,
    "max_features": 0.1,
    "min_samples_leaf": 1,
    "max_depth": None,
    "random_state": 42,
    "n_jobs": -1,
}
TUNING_REPORT_PATH = REPORT_DIR / "rf_tuning_results.json"


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """计算回归模型的 RMSE、MAE 和 R²。"""
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def load_split(path: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """读取一个固定划分文件并生成特征矩阵。"""
    if not path.exists():
        raise FileNotFoundError(f"找不到数据文件：{path}")

    df = pd.read_csv(path)
    missing_columns = {"smiles", "exp"} - set(df.columns)
    if missing_columns:
        raise ValueError(f"{path.name} 缺少必要列：{sorted(missing_columns)}")

    labels = pd.to_numeric(df["exp"], errors="coerce")
    if labels.isna().any():
        bad_rows = (labels[labels.isna()].index + 2).tolist()
        raise ValueError(f"{path.name} 的 exp 无效，CSV 行号：{bad_rows}")

    features, invalid_positions = smiles_list_to_matrix(
        df["smiles"].tolist(),
        radius=FEATURE_CONFIG["radius"],
        fp_size=FEATURE_CONFIG["fp_size"],
    )
    if invalid_positions:
        bad_rows = [position + 2 for position in invalid_positions]
        raise ValueError(f"{path.name} 包含无效 SMILES，CSV 行号：{bad_rows}")

    return df, features, labels.to_numpy(dtype=float)


def build_model() -> RandomForestRegressor:
    """用统一参数创建随机森林。"""
    return RandomForestRegressor(**MODEL_PARAMS)


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("读取固定 Scaffold Split 并生成 Morgan 指纹...", flush=True)
    train_df, x_train, y_train = load_split(TRAIN_PATH)
    valid_df, x_valid, y_valid = load_split(VALID_PATH)
    print(f"训练集 {len(train_df)} 条，验证集 {len(valid_df)} 条")

    # 均值基线只使用训练标签，作为模型效果的最低参照。
    train_mean = float(np.mean(y_train))
    baseline_train_pred = np.full_like(y_train, train_mean, dtype=float)
    baseline_valid_pred = np.full_like(y_valid, train_mean, dtype=float)

    # 候选模型只使用训练集拟合，验证集用于评估既定基线参数。
    print("训练 Random Forest 基线模型...", flush=True)
    validation_model = build_model()
    validation_model.fit(x_train, y_train)
    train_pred = validation_model.predict(x_train)
    valid_pred = validation_model.predict(x_valid)

    train_metrics = calculate_metrics(y_train, train_pred)
    valid_metrics = calculate_metrics(y_valid, valid_pred)
    baseline_train_metrics = calculate_metrics(y_train, baseline_train_pred)
    baseline_valid_metrics = calculate_metrics(y_valid, baseline_valid_pred)

    # 参数确定后，用训练集和验证集重新训练最终模型。内部测试集直到此时
    # 才首次参与评估，并且不会用于修改模型参数。
    print("使用训练集和验证集拟合最终模型...", flush=True)
    x_final = np.concatenate([x_train, x_valid], axis=0)
    y_final = np.concatenate([y_train, y_valid], axis=0)
    final_model = build_model()
    final_model.fit(x_final, y_final)

    print("最终模型已确定，开始读取内部测试集...", flush=True)
    test_df, x_test, y_test = load_split(TEST_PATH)
    test_pred = final_model.predict(x_test)
    test_metrics = calculate_metrics(y_test, test_pred)
    final_mean = float(np.mean(y_final))
    baseline_test_pred = np.full_like(y_test, final_mean, dtype=float)
    baseline_test_metrics = calculate_metrics(y_test, baseline_test_pred)

    model_bundle = {
        "model": final_model,
        "feature_config": FEATURE_CONFIG,
        "model_params": MODEL_PARAMS,
        "parameter_selection": "lowest validation RMSE in rf_tuning_results.json",
        "target_column": "exp",
        "training_files": [TRAIN_PATH.name, VALID_PATH.name],
        "internal_test_file": TEST_PATH.name,
    }
    joblib.dump(model_bundle, MODEL_PATH)

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
        "feature_config": FEATURE_CONFIG,
        "model_params": MODEL_PARAMS,
        "parameter_selection": {
            "metric": "validation RMSE",
            "tuning_report": TUNING_REPORT_PATH.name,
            "internal_test_used_for_selection": False,
        },
        "mean_baseline": {
            "train": baseline_train_metrics,
            "validation": baseline_valid_metrics,
            "internal_test": baseline_test_metrics,
        },
        "random_forest": {
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
        },
    }
    with REPORT_PATH.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    print("\n===== 模型评估结果 =====")
    print("训练集（重新拟合前）：", train_metrics)
    print("验证集（重新拟合前）：", valid_metrics)
    print("内部测试集（最终模型）：", test_metrics)
    print("内部测试集均值基线：", baseline_test_metrics)
    print(f"模型已保存：{MODEL_PATH.relative_to(PROJECT_DIR)}")
    print(f"报告已保存：{REPORT_PATH.relative_to(PROJECT_DIR)}")
    print(
        "内部测试预测已保存："
        f"{TEST_PREDICTIONS_PATH.relative_to(PROJECT_DIR)}"
    )


if __name__ == "__main__":
    main()

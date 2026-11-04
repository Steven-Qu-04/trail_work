"""加载已训练模型，对外部 CSV 预测并在有标签时计算指标。"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from features import smiles_list_to_feature_matrix


PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = PROJECT_DIR / "models" / "production" / "xgboost_svr_ensemble.joblib"
DEFAULT_INPUT_PATH = PROJECT_DIR / "data" / "processed" / "scaffold_test.csv"
DEFAULT_OUTPUT_PATH = PROJECT_DIR / "reports" / "production" / "test_predictions.csv"
DEFAULT_REPORT_PATH = PROJECT_DIR / "reports" / "production" / "test_report.json"


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """计算外部测试集的回归指标。"""
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def resolve_path(path_text: str) -> Path:
    """将相对路径解释为相对于项目根目录的路径。"""
    path = Path(path_text)
    return path if path.is_absolute() else PROJECT_DIR / path


def main() -> None:
    parser = argparse.ArgumentParser(description="预测外部分子的 logD")
    parser.add_argument("--input", default=str(DEFAULT_INPUT_PATH))
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--report", default=str(DEFAULT_REPORT_PATH))
    parser.add_argument(
        "--require-labels",
        action="store_true",
        help="要求输入包含至少2条有效exp标签；用于正式带标签测试集评估",
    )
    args = parser.parse_args()

    input_path = resolve_path(args.input)
    model_path = resolve_path(args.model)
    output_path = resolve_path(args.output)
    report_path = resolve_path(args.report)

    if not input_path.exists():
        raise FileNotFoundError(f"找不到输入文件：{input_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"找不到模型文件：{model_path}")

    df = pd.read_csv(input_path)
    if "smiles" not in df.columns:
        raise ValueError("测试文件缺少必要字段：smiles")
    if args.require_labels and "exp" not in df.columns:
        raise ValueError("带标签评估要求测试文件包含字段：exp")

    bundle = joblib.load(model_path)
    is_ensemble = bundle.get("model_type") == "WeightedEnsemble"
    if is_ensemble:
        if not bundle.get("models"):
            raise ValueError("融合模型中没有子模型")
        feature_config = [item["feature_config"] for item in bundle["models"]]
        component_features, invalid_sets = [], []
        for item in bundle["models"]:
            matrix, invalid = smiles_list_to_feature_matrix(
                df["smiles"].tolist(), feature_config=item["feature_config"])
            component_features.append(matrix)
            invalid_sets.append(set(invalid))
        invalid_positions = sorted(set().union(*invalid_sets))
    else:
        if "model" not in bundle or "feature_config" not in bundle:
            raise ValueError("模型文件缺少 model 或 feature_config")
        feature_config = bundle["feature_config"]
        features, invalid_positions = smiles_list_to_feature_matrix(
            df["smiles"].tolist(), feature_config=feature_config)
    valid_mask = np.ones(len(df), dtype=bool)
    valid_mask[invalid_positions] = False
    valid_positions = np.flatnonzero(valid_mask)

    if len(valid_positions) == 0:
        raise ValueError("测试文件中没有可用于预测的有效 SMILES")

    if is_ensemble:
        predictions = sum(
            item["weight"] * item["model"].predict(matrix)
            for item, matrix in zip(bundle["models"], component_features)
        )
    else:
        predictions = bundle["model"].predict(features)
    result_df = df.copy()
    result_df["predicted_exp"] = np.nan
    result_df["prediction_status"] = "invalid_smiles"
    result_df.loc[valid_positions, "predicted_exp"] = predictions
    result_df.loc[valid_positions, "prediction_status"] = "ok"

    metrics = None
    error_analysis = None
    evaluated_rows = 0
    if "exp" in df.columns:
        labels = pd.to_numeric(df.loc[valid_positions, "exp"], errors="coerce")
        labeled_mask = labels.notna().to_numpy()
        evaluated_rows = int(labeled_mask.sum())
        if evaluated_rows >= 2:
            y_true = labels.to_numpy(dtype=float)[labeled_mask]
            y_pred = predictions[labeled_mask]
            metrics = calculate_metrics(y_true, y_pred)
            evaluated_positions = valid_positions[labeled_mask]
            result_df["residual"] = np.nan
            result_df["absolute_error"] = np.nan
            result_df.loc[evaluated_positions, "residual"] = y_true - y_pred
            result_df.loc[evaluated_positions, "absolute_error"] = np.abs(
                y_true - y_pred
            )
            absolute_errors = np.abs(y_true - y_pred)
            error_analysis = {
                "mean_true_exp": float(np.mean(y_true)),
                "mean_predicted_exp": float(np.mean(y_pred)),
                "max_absolute_error": float(np.max(absolute_errors)),
                "absolute_error_gt_0_5": int(np.sum(absolute_errors > 0.5)),
                "absolute_error_gt_1_0": int(np.sum(absolute_errors > 1.0)),
                "absolute_error_gt_2_0": int(np.sum(absolute_errors > 2.0)),
            }

    if args.require_labels and evaluated_rows < 2:
        raise ValueError(
            "带标签评估至少需要2条同时具有有效SMILES和数值型exp的记录"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_path, index=False)

    report = {
        "input_file": str(input_path),
        "model_file": str(model_path),
        "output_file": str(output_path),
        "model_type": bundle.get("model_type", type(bundle.get("model")).__name__),
        "target_column": "exp",
        "prediction_column": "predicted_exp",
        "require_labels": args.require_labels,
        "total_rows": len(df),
        "valid_predictions": len(valid_positions),
        "invalid_smiles_rows": [position + 2 for position in invalid_positions],
        "evaluated_rows": evaluated_rows,
        "feature_config": feature_config,
        "metrics": metrics,
        "error_analysis": error_analysis,
    }
    if metrics is None:
        report["metrics_message"] = "有效 exp 少于 2 条，未计算完整评估指标"

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    print("预测完成！")
    print(f"有效预测：{len(valid_positions)}/{len(df)}")
    print(f"预测结果：{output_path}")
    print(f"检验报告：{report_path}")
    if metrics is not None:
        print("评估指标：", metrics)


if __name__ == "__main__":
    main()

"""依次执行特征消融、指纹比较和 XGBoost 精细调参。"""

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
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor

from features import DEFAULT_DESCRIPTOR_NAMES, smiles_list_to_feature_matrix
from train import (
    COMPONENT_MODEL_DIR,
    PROJECT_DIR,
    XGBOOST_REPORT_DIR,
    TEST_PATH,
    TRAIN_PATH,
    VALID_PATH,
    calculate_metrics,
)
from train_xgboost_descriptors import scaffold_from_smiles


MODEL_PATH = COMPONENT_MODEL_DIR / "xgboost_optimized_model.joblib"
REPORT_PATH = XGBOOST_REPORT_DIR / "xgboost_optimized_metrics.json"
FEATURE_RESULTS_CSV = XGBOOST_REPORT_DIR / "xgboost_feature_experiments.csv"
FEATURE_RESULTS_JSON = XGBOOST_REPORT_DIR / "xgboost_feature_experiments.json"
PARAM_RESULTS_CSV = XGBOOST_REPORT_DIR / "xgboost_optimized_cv_results.csv"
PARAM_RESULTS_JSON = XGBOOST_REPORT_DIR / "xgboost_optimized_cv_results.json"
OOF_PATH = XGBOOST_REPORT_DIR / "xgboost_optimized_oof_predictions.csv"
IMPORTANCE_PATH = XGBOOST_REPORT_DIR / "xgboost_optimized_feature_importance.csv"
TEST_PREDICTIONS_PATH = XGBOOST_REPORT_DIR / "xgboost_optimized_test_predictions.csv"
LARGEST_ERRORS_PATH = XGBOOST_REPORT_DIR / "xgboost_optimized_largest_errors.csv"

ALL_DESCRIPTORS = list(DEFAULT_DESCRIPTOR_NAMES)
NO_SIZE_DESCRIPTORS = [
    name for name in ALL_DESCRIPTORS if name not in {"MolWt", "HeavyAtomCount"}
]
NO_POLARITY_DESCRIPTORS = [
    name
    for name in ALL_DESCRIPTORS
    if name not in {"TPSA", "NumHDonors", "NumHAcceptors", "FormalCharge"}
]

FEATURE_EXPERIMENTS = [
    {"name": "MorganBit_r2_2048", "type": "Morgan", "radius": 2, "fp_size": 2048},
    {"name": "Descriptors_only", "type": "RDKitDescriptors", "descriptor_names": ALL_DESCRIPTORS},
    {"name": "MorganBit_r2_2048_all_desc", "type": "Morgan+RDKitDescriptors", "radius": 2, "fp_size": 2048, "descriptor_names": ALL_DESCRIPTORS},
    {"name": "MorganBit_r2_2048_no_size", "type": "Morgan+RDKitDescriptors", "radius": 2, "fp_size": 2048, "descriptor_names": NO_SIZE_DESCRIPTORS},
    {"name": "MorganBit_r2_2048_no_polarity", "type": "Morgan+RDKitDescriptors", "radius": 2, "fp_size": 2048, "descriptor_names": NO_POLARITY_DESCRIPTORS},
    {"name": "MorganBit_r3_2048_all_desc", "type": "Morgan+RDKitDescriptors", "radius": 3, "fp_size": 2048, "descriptor_names": ALL_DESCRIPTORS},
    {"name": "MorganBit_r2_4096_all_desc", "type": "Morgan+RDKitDescriptors", "radius": 2, "fp_size": 4096, "descriptor_names": ALL_DESCRIPTORS},
    {"name": "MorganBit_r3_4096_all_desc", "type": "Morgan+RDKitDescriptors", "radius": 3, "fp_size": 4096, "descriptor_names": ALL_DESCRIPTORS},
    {"name": "MorganCount_r2_2048_all_desc", "type": "MorganCount+RDKitDescriptors", "radius": 2, "fp_size": 2048, "descriptor_names": ALL_DESCRIPTORS},
    {"name": "MorganCount_r3_2048_all_desc", "type": "MorganCount+RDKitDescriptors", "radius": 3, "fp_size": 2048, "descriptor_names": ALL_DESCRIPTORS},
]

COMMON_PARAMS = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "tree_method": "hist",
    "n_estimators": 1800,
    "early_stopping_rounds": 50,
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": 0,
}

SCREEN_PARAMS = {
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.6,
    "min_child_weight": 1,
    "reg_lambda": 5.0,
    "reg_alpha": 0.5,
    "gamma": 0.0,
}

PARAM_CANDIDATES = [
    {**SCREEN_PARAMS},
    {**SCREEN_PARAMS, "reg_alpha": 1.0},
    {**SCREEN_PARAMS, "reg_lambda": 10.0},
    {**SCREEN_PARAMS, "max_depth": 4},
    {**SCREEN_PARAMS, "max_depth": 6},
    {**SCREEN_PARAMS, "min_child_weight": 3},
    {**SCREEN_PARAMS, "gamma": 0.05},
]


def load_raw_split(path):
    df = pd.read_csv(path)
    labels = pd.to_numeric(df["exp"], errors="coerce")
    if labels.isna().any():
        raise ValueError(f"{path.name} 包含无效 exp")
    return df, labels.to_numpy(dtype=float)


def make_features(df, feature_config):
    config = {key: value for key, value in feature_config.items() if key != "name"}
    matrix, invalid_positions = smiles_list_to_feature_matrix(
        df["smiles"].tolist(),
        config,
    )
    if invalid_positions:
        raise ValueError(f"检测到无效 SMILES 位置：{invalid_positions}")
    return matrix


def evaluate_cv(x, y, folds, params, keep_oof=False):
    fold_metrics = []
    tree_counts = []
    oof_predictions = np.full(len(y), np.nan, dtype=float)

    for fit_indices, valid_indices in folds:
        model = XGBRegressor(**{**COMMON_PARAMS, **params})
        model.fit(
            x[fit_indices],
            y[fit_indices],
            eval_set=[(x[valid_indices], y[valid_indices])],
            verbose=False,
        )
        predictions = model.predict(x[valid_indices])
        fold_metrics.append(calculate_metrics(y[valid_indices], predictions))
        tree_counts.append(int(model.best_iteration) + 1)
        if keep_oof:
            oof_predictions[valid_indices] = predictions

    result = {
        "cv_mean_RMSE": float(np.mean([m["RMSE"] for m in fold_metrics])),
        "cv_std_RMSE": float(np.std([m["RMSE"] for m in fold_metrics])),
        "cv_mean_MAE": float(np.mean([m["MAE"] for m in fold_metrics])),
        "cv_mean_R2": float(np.mean([m["R2"] for m in fold_metrics])),
        "median_n_estimators": int(round(statistics.median(tree_counts))),
        "fold_RMSE": [m["RMSE"] for m in fold_metrics],
        "fold_n_estimators": tree_counts,
    }
    return result, oof_predictions


def feature_names(config):
    kind = config["type"]
    descriptors = list(config.get("descriptor_names", []))
    names = []
    if kind != "RDKitDescriptors":
        prefix = "MorganCount" if kind.startswith("MorganCount") else "MorganBit"
        names.extend([f"{prefix}_{i}" for i in range(config["fp_size"])])
    if kind.endswith("RDKitDescriptors") or kind == "RDKitDescriptors":
        names.extend(descriptors)
    return names


def main() -> None:
    COMPONENT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    XGBOOST_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    train_df, y_train = load_raw_split(TRAIN_PATH)
    groups = np.asarray(
        [scaffold_from_smiles(smiles) for smiles in train_df["smiles"]],
        dtype=object,
    )
    splitter = GroupKFold(n_splits=5, shuffle=True, random_state=42)
    folds = list(splitter.split(train_df, y_train, groups=groups))

    print("===== 第一阶段：特征消融与指纹比较 =====", flush=True)
    feature_results = []
    for number, config in enumerate(FEATURE_EXPERIMENTS, start=1):
        print(f"[{number}/{len(FEATURE_EXPERIMENTS)}] {config['name']}", flush=True)
        started = time.perf_counter()
        x_train = make_features(train_df, config)
        cv_result, _ = evaluate_cv(x_train, y_train, folds, SCREEN_PARAMS)
        result = {
            "feature_name": config["name"],
            "feature_type": config["type"],
            "feature_count": x_train.shape[1],
            **cv_result,
            "elapsed_seconds": time.perf_counter() - started,
            "feature_config": config,
        }
        feature_results.append(result)
        print(
            f"    RMSE={result['cv_mean_RMSE']:.6f}±"
            f"{result['cv_std_RMSE']:.6f}, R²={result['cv_mean_R2']:.6f}",
            flush=True,
        )

    feature_results.sort(key=lambda item: item["cv_mean_RMSE"])
    best_feature = feature_results[0]["feature_config"]
    feature_csv_rows = [
        {key: value for key, value in item.items() if not isinstance(value, (list, dict))}
        for item in feature_results
    ]
    pd.DataFrame(feature_csv_rows).to_csv(FEATURE_RESULTS_CSV, index=False)
    with FEATURE_RESULTS_JSON.open("w", encoding="utf-8") as file:
        json.dump(
            {"results": feature_results, "best_feature_config": best_feature},
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(f"最佳特征：{best_feature['name']}", flush=True)
    x_train = make_features(train_df, best_feature)

    print("\n===== 第二阶段：最佳特征的小范围参数调节 =====", flush=True)
    param_results = []
    oof_by_candidate = {}
    for number, params in enumerate(PARAM_CANDIDATES, start=1):
        print(f"[{number}/{len(PARAM_CANDIDATES)}] {params}", flush=True)
        started = time.perf_counter()
        result, oof_predictions = evaluate_cv(
            x_train,
            y_train,
            folds,
            params,
            keep_oof=True,
        )
        param_results.append(
            {
                "candidate": number,
                **params,
                **result,
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
        oof_by_candidate[number] = oof_predictions
        print(
            f"    RMSE={result['cv_mean_RMSE']:.6f}±"
            f"{result['cv_std_RMSE']:.6f}, R²={result['cv_mean_R2']:.6f}",
            flush=True,
        )

    param_results.sort(key=lambda item: item["cv_mean_RMSE"])
    best_result = param_results[0]
    best_candidate = best_result["candidate"]
    best_params = {
        "objective": COMMON_PARAMS["objective"],
        "eval_metric": COMMON_PARAMS["eval_metric"],
        "tree_method": COMMON_PARAMS["tree_method"],
        "n_estimators": best_result["median_n_estimators"],
        "random_state": COMMON_PARAMS["random_state"],
        "n_jobs": COMMON_PARAMS["n_jobs"],
        "verbosity": COMMON_PARAMS["verbosity"],
        **{key: best_result[key] for key in SCREEN_PARAMS},
    }
    param_csv_rows = [
        {key: value for key, value in item.items() if not isinstance(value, list)}
        for item in param_results
    ]
    pd.DataFrame(param_csv_rows).to_csv(PARAM_RESULTS_CSV, index=False)
    with PARAM_RESULTS_JSON.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "feature_config": best_feature,
                "results": param_results,
                "best_params": best_params,
            },
            file,
            indent=2,
            ensure_ascii=False,
        )

    oof_predictions = oof_by_candidate[best_candidate]
    pd.DataFrame(
        {
            "row_index": np.arange(len(train_df)),
            "smiles": train_df["smiles"],
            "exp": y_train,
            "xgboost_oof_prediction": oof_predictions,
        }
    ).to_csv(OOF_PATH, index=False)

    print("\n===== 第三阶段：独立验证与最终测试 =====", flush=True)
    valid_df, y_valid = load_raw_split(VALID_PATH)
    x_valid = make_features(valid_df, best_feature)
    validation_model = XGBRegressor(**best_params)
    validation_model.fit(x_train, y_train, verbose=False)
    train_metrics = calculate_metrics(y_train, validation_model.predict(x_train))
    valid_metrics = calculate_metrics(y_valid, validation_model.predict(x_valid))

    x_final = np.concatenate([x_train, x_valid], axis=0)
    y_final = np.concatenate([y_train, y_valid], axis=0)
    final_model = XGBRegressor(**best_params)
    final_model.fit(x_final, y_final, verbose=False)

    test_df, y_test = load_raw_split(TEST_PATH)
    x_test = make_features(test_df, best_feature)
    test_predictions = final_model.predict(x_test)
    test_metrics = calculate_metrics(y_test, test_predictions)

    clean_feature_config = {
        key: value for key, value in best_feature.items() if key != "name"
    }
    bundle = {
        "model": final_model,
        "model_type": "XGBRegressor",
        "feature_config": clean_feature_config,
        "model_params": best_params,
        "parameter_selection": "5-fold scaffold CV on train.csv",
        "target_column": "exp",
        "training_files": [TRAIN_PATH.name, VALID_PATH.name],
        "internal_test_file": TEST_PATH.name,
    }
    joblib.dump(bundle, MODEL_PATH, compress=3)

    names = feature_names(best_feature)
    importance_df = pd.DataFrame(
        {"feature": names, "importance": final_model.feature_importances_}
    ).sort_values("importance", ascending=False)
    importance_df.to_csv(IMPORTANCE_PATH, index=False)

    test_results = test_df.copy()
    test_results["predicted_exp"] = test_predictions
    test_results["residual"] = y_test - test_predictions
    test_results["absolute_error"] = np.abs(test_results["residual"])
    test_results.to_csv(TEST_PREDICTIONS_PATH, index=False)
    test_results.nlargest(20, "absolute_error").to_csv(
        LARGEST_ERRORS_PATH,
        index=False,
    )

    report = {
        "best_feature_config": clean_feature_config,
        "best_model_params": best_params,
        "scaffold_cv": {
            "mean_RMSE": best_result["cv_mean_RMSE"],
            "std_RMSE": best_result["cv_std_RMSE"],
            "mean_MAE": best_result["cv_mean_MAE"],
            "mean_R2": best_result["cv_mean_R2"],
        },
        "metrics": {
            "train_before_refit": train_metrics,
            "validation_before_refit": valid_metrics,
            "internal_test_after_refit": test_metrics,
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

    print("\n===== 优化完成 =====")
    print("最佳特征：", clean_feature_config)
    print("CV：", report["scaffold_cv"])
    print("验证集：", valid_metrics)
    print("内部测试集：", test_metrics)
    print(f"模型：{MODEL_PATH.relative_to(PROJECT_DIR)}")


if __name__ == "__main__":
    main()

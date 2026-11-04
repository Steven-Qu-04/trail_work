"""Evaluate an XGBoost/SVR ensemble without using the test set for selection."""

import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.svm import SVR
from xgboost import XGBRegressor

from features import smiles_list_to_feature_matrix
from train import (
    COMPONENT_MODEL_DIR,
    PRODUCTION_MODEL_DIR,
    PRODUCTION_REPORT_DIR,
    TEST_PATH,
    TRAIN_PATH,
    VALID_PATH,
    calculate_metrics,
)
from train_xgboost_descriptors import scaffold_from_smiles

XGB_PATH = COMPONENT_MODEL_DIR / "xgboost_optimized_model.joblib"
SVR_PATH = COMPONENT_MODEL_DIR / "svr_model.joblib"
MODEL_PATH = PRODUCTION_MODEL_DIR / "xgboost_svr_ensemble.joblib"
OOF_PATH = PRODUCTION_REPORT_DIR / "ensemble_oof_predictions.csv"
WEIGHTS_PATH = PRODUCTION_REPORT_DIR / "ensemble_weight_search.csv"
PREDICTIONS_PATH = PRODUCTION_REPORT_DIR / "ensemble_test_predictions.csv"
REPORT_PATH = PRODUCTION_REPORT_DIR / "ensemble_metrics.json"


def load_data(path):
    df = pd.read_csv(path)
    return df, pd.to_numeric(df["exp"], errors="raise").to_numpy(float)


def make_features(df, config):
    matrix, invalid = smiles_list_to_feature_matrix(df["smiles"].tolist(), config)
    if invalid:
        raise ValueError(f"Invalid SMILES positions: {invalid}")
    return matrix


def main():
    PRODUCTION_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    PRODUCTION_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    xgb_bundle, svr_bundle = joblib.load(XGB_PATH), joblib.load(SVR_PATH)
    train_df, y_train = load_data(TRAIN_PATH)
    valid_df, y_valid = load_data(VALID_PATH)
    test_df, y_test = load_data(TEST_PATH)
    train_xgb = make_features(train_df, xgb_bundle["feature_config"])
    valid_xgb = make_features(valid_df, xgb_bundle["feature_config"])
    test_xgb = make_features(test_df, xgb_bundle["feature_config"])
    train_svr = make_features(train_df, svr_bundle["feature_config"])
    valid_svr = make_features(valid_df, svr_bundle["feature_config"])
    test_svr = make_features(test_df, svr_bundle["feature_config"])

    groups = np.asarray([scaffold_from_smiles(s) for s in train_df["smiles"]], dtype=object)
    folds = GroupKFold(n_splits=5, shuffle=True, random_state=42).split(train_df, y_train, groups)
    xgb_oof, svr_oof = np.full(len(train_df), np.nan), np.full(len(train_df), np.nan)
    for fold, (fit_idx, holdout_idx) in enumerate(folds, start=1):
        print(f"OOF fold {fold}/5", flush=True)
        xgb = XGBRegressor(**xgb_bundle["model_params"]).fit(
            train_xgb[fit_idx], y_train[fit_idx], verbose=False)
        svr = SVR(**svr_bundle["model_params"]).fit(train_svr[fit_idx], y_train[fit_idx])
        xgb_oof[holdout_idx] = xgb.predict(train_xgb[holdout_idx])
        svr_oof[holdout_idx] = svr.predict(train_svr[holdout_idx])

    searches = []
    for xgb_weight in np.linspace(0, 1, 21):
        pred = xgb_weight * xgb_oof + (1 - xgb_weight) * svr_oof
        searches.append({"xgboost_weight": float(xgb_weight), "svr_weight": float(1-xgb_weight),
                         **calculate_metrics(y_train, pred)})
    searches.sort(key=lambda x: x["RMSE"])
    best = searches[0]
    pd.DataFrame(searches).to_csv(WEIGHTS_PATH, index=False)
    ensemble_oof = best["xgboost_weight"] * xgb_oof + best["svr_weight"] * svr_oof
    pd.DataFrame({"row_index": np.arange(len(train_df)), "smiles": train_df["smiles"],
                  "exp": y_train, "xgboost_oof_prediction": xgb_oof,
                  "svr_oof_prediction": svr_oof,
                  "ensemble_oof_prediction": ensemble_oof}).to_csv(OOF_PATH, index=False)

    xgb_train_model = XGBRegressor(**xgb_bundle["model_params"]).fit(train_xgb, y_train, verbose=False)
    svr_train_model = SVR(**svr_bundle["model_params"]).fit(train_svr, y_train)
    xgb_valid_pred, svr_valid_pred = xgb_train_model.predict(valid_xgb), svr_train_model.predict(valid_svr)
    ensemble_valid = best["xgboost_weight"]*xgb_valid_pred + best["svr_weight"]*svr_valid_pred
    xgb_test_pred = xgb_bundle["model"].predict(test_xgb)
    svr_test_pred = svr_bundle["model"].predict(test_svr)
    ensemble_test = best["xgboost_weight"]*xgb_test_pred + best["svr_weight"]*svr_test_pred

    results = test_df.copy()
    results["xgboost_prediction"], results["svr_prediction"] = xgb_test_pred, svr_test_pred
    results["predicted_exp"] = ensemble_test
    results["residual"] = y_test - ensemble_test
    results["absolute_error"] = np.abs(results["residual"])
    results.to_csv(PREDICTIONS_PATH, index=False)

    bundle = {"model_type": "WeightedEnsemble", "models": [
        {"name": "XGBRegressor", "model": xgb_bundle["model"],
         "feature_config": xgb_bundle["feature_config"], "weight": best["xgboost_weight"]},
        {"name": "SVR", "model": svr_bundle["model"],
         "feature_config": svr_bundle["feature_config"], "weight": best["svr_weight"]}],
        "weight_selection": "lowest OOF RMSE from 5-fold scaffold CV on train.csv",
        "training_files": [TRAIN_PATH.name, VALID_PATH.name]}
    joblib.dump(bundle, MODEL_PATH, compress=3)
    report = {
        "selection": {"method": bundle["weight_selection"],
                      "xgboost_weight": best["xgboost_weight"], "svr_weight": best["svr_weight"],
                      "internal_test_used_for_selection": False},
        "oof_train": {"xgboost": calculate_metrics(y_train, xgb_oof),
                      "svr": calculate_metrics(y_train, svr_oof),
                      "ensemble": calculate_metrics(y_train, ensemble_oof)},
        "validation_before_refit": {"xgboost": calculate_metrics(y_valid, xgb_valid_pred),
                                    "svr": calculate_metrics(y_valid, svr_valid_pred),
                                    "ensemble": calculate_metrics(y_valid, ensemble_valid)},
        "internal_test_after_refit": {"xgboost": calculate_metrics(y_test, xgb_test_pred),
                                      "svr": calculate_metrics(y_test, svr_test_pred),
                                      "ensemble": calculate_metrics(y_test, ensemble_test)}}
    with REPORT_PATH.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

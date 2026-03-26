import json
import os
import joblib
import numpy as np
import pandas as pd

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, average_precision_score
from sklearn.calibration import CalibratedClassifierCV

try:
    from sklearn.frozen import FrozenEstimator
except ImportError:
    FrozenEstimator = None


DATA_PATH = "data/training_data_severity.csv"
MODEL_PATH = "artifacts/safesite_xgb_weighted_monotonic.joblib"
METADATA_PATH = "artifacts/model_metadata.json"
FEATURE_IMPORTANCE_PATH = "artifacts/feature_importance.csv"

os.makedirs("artifacts", exist_ok=True)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["total_deaths"] = pd.to_numeric(df["total_deaths"], errors="coerce").fillna(0).clip(lower=0)
    df["total_hours_worked"] = pd.to_numeric(df["total_hours_worked"], errors="coerce").fillna(0).clip(lower=0)

    df["death_flag"] = (df["total_deaths"] > 0).astype(int)
    df["total_deaths_log"] = np.log1p(df["total_deaths"])

    # Stronger fatality representation
    df["death_severity"] = df["death_flag"] * (1.0 + np.clip(df["total_deaths_log"], 0, 2.0))

    # Exposure-adjusted fatality proxy
    df["death_rate_proxy"] = df["total_deaths"] / (df["total_hours_worked"] + 1.0)
    df["death_rate_proxy_log"] = np.log1p(df["death_rate_proxy"] * 200000.0)

    return df


def make_serializable_thresholds(thresholds: dict) -> dict:
    serializable = {}
    for band, bounds in thresholds.items():
        serializable[band] = [float(bounds[0]), float(bounds[1])]
    return serializable


def assign_risk_band(prob: float, thresholds: dict) -> str:
    if prob < thresholds["low"][1]:
        return "Low"
    if prob < thresholds["medium"][1]:
        return "Medium"
    if prob < thresholds["high"][1]:
        return "High"
    return "Critical"


def main():
    df = pd.read_csv(DATA_PATH)
    df = build_features(df)

    FEATURE_COLS = [
        "annual_average_employees",
        "total_hours_worked",
        "death_flag",
        "total_deaths_log",
        "death_severity",
        "death_rate_proxy_log",
        "total_dafw_cases",
        "total_djtr_cases",
        "total_other_cases",
        "total_dafw_days",
        "total_djtr_days",
        "tcr_rate_log",
        "dart_rate_log",
        "dafw_rate_log",
    ]
    TARGET_COL = "high_risk"

    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL]).copy()
    df[TARGET_COL] = df[TARGET_COL].astype(int)

    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL].copy()

    # Split 1: hold-out test set
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    # Split 2: separate calibration set
    X_train, X_cal, y_train, y_cal = train_test_split(
        X_train_full,
        y_train_full,
        test_size=0.25,   # 25% of 80% = 20% overall
        random_state=42,
        stratify=y_train_full,
    )

    # Class imbalance
    pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()

    # Sample weights: force the learner to pay more attention to fatal cases
    sample_weights = np.ones(len(X_train), dtype=float)
    fatal_mask = X_train["death_flag"].to_numpy() == 1
    sample_weights[fatal_mask] = 12.0

    base_model = XGBClassifier(
        n_estimators=450,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=pos_weight,
        monotone_constraints=(0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
        random_state=42,
        eval_metric="logloss",
    )

    # Fit the weighted base model once
    base_model.fit(X_train, y_train, sample_weight=sample_weights)

    # Feature importance from the actually fitted weighted model
    feature_importance_df = pd.DataFrame(
        {
            "feature": FEATURE_COLS,
            "importance": base_model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    feature_importance_df.to_csv(FEATURE_IMPORTANCE_PATH, index=False)

    # Proper post-fit calibration on a disjoint calibration split
    if FrozenEstimator is not None:
        calibrated_model = CalibratedClassifierCV(
            estimator=FrozenEstimator(base_model),
            method="sigmoid",
            cv=None,
        )
        calibrated_model.fit(X_cal, y_cal)
    else:
        # Conservative fallback for older sklearn versions
        calibrated_model = CalibratedClassifierCV(
            estimator=base_model,
            method="sigmoid",
            cv=3,
        )
        calibrated_model.fit(X_train_full, y_train_full)

    # Evaluate on untouched test set
    y_probs = calibrated_model.predict_proba(X_test)[:, 1]

    threshold = 0.30
    y_pred = (y_probs >= threshold).astype(int)

    pr_auc = average_precision_score(y_test, y_probs)

    print("\nPR-AUC:", pr_auc)

    print("\nCLASSIFICATION REPORT:")
    print(classification_report(y_test, y_pred))

    print("\nCONFUSION MATRIX:")
    print(confusion_matrix(y_test, y_pred))

    positive_probs = y_probs[y_probs >= threshold]

    if len(positive_probs) == 0:
        q70 = threshold
        q90 = threshold
    else:
        q70 = float(np.quantile(positive_probs, 0.70))
        q90 = float(np.quantile(positive_probs, 0.90))

        if q70 < threshold:
            q70 = threshold
        if q90 < q70:
            q90 = q70

    risk_band_thresholds = {
        "low": [0.0, float(threshold)],
        "medium": [float(threshold), float(q70)],
        "high": [float(q70), float(q90)],
        "critical": [float(q90), 1.0],
    }

    print("\nRISK BAND THRESHOLDS:")
    for band, bounds in risk_band_thresholds.items():
        print(f"{band}: {bounds}")

    sample_predictions = pd.DataFrame(
        {
            "probability": y_probs[:10],
            "predicted_class": y_pred[:10],
            "risk_band": [assign_risk_band(float(p), risk_band_thresholds) for p in y_probs[:10]],
        }
    )

    print("\nSAMPLE PREDICTIONS:")
    print(sample_predictions)

    # Save backward-compatible bundle for app.py / validation runner
    model_bundle = {
        "model": calibrated_model,
        "feature_cols": FEATURE_COLS,
        "threshold": float(threshold),
        "risk_band_thresholds": make_serializable_thresholds(risk_band_thresholds),
    }

    joblib.dump(model_bundle, MODEL_PATH)

    metadata = {
        "feature_cols": FEATURE_COLS,
        "threshold": float(threshold),
        "risk_band_thresholds": make_serializable_thresholds(risk_band_thresholds),
    }

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    print("\nTOP FEATURE IMPORTANCE:")
    print(feature_importance_df.head(15))

    print(f"\nMODEL SAVED TO: {MODEL_PATH}")
    print(f"METADATA SAVED TO: {METADATA_PATH}")
    print(f"FEATURE IMPORTANCE SAVED TO: {FEATURE_IMPORTANCE_PATH}")


if __name__ == "__main__":
    main()
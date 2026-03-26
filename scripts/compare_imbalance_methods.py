import pandas as pd
from sklearn.metrics import average_precision_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.combine import SMOTETomek

DATA_PATH = "data/training_data_severity.csv"
TARGET_COL = "high_risk"
THRESHOLD = 0.30
RANDOM_STATE = 42

FEATURE_COLS = [
    "annual_average_employees",
    "total_hours_worked",
    "total_injuries",
    "total_deaths",
    "total_dafw_cases",
    "total_djtr_cases",
    "total_other_cases",
    "total_dafw_days",
    "total_djtr_days",
    "incident_rate_log",
    "tcr_rate_log",
    "dart_rate_log",
    "dafw_rate_log"
]


def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=[TARGET_COL]).copy()
    df[TARGET_COL] = df[TARGET_COL].astype(int)
    return df


def split_data(df: pd.DataFrame):
    train_df = df[df["year_filing_for"] <= 2021].copy()
    test_df = df[df["year_filing_for"] == 2022].copy()

    X_train = train_df[FEATURE_COLS].fillna(0)
    y_train = train_df[TARGET_COL].astype(int)

    X_test = test_df[FEATURE_COLS].fillna(0)
    y_test = test_df[TARGET_COL].astype(int)

    return X_train, y_train, X_test, y_test


def build_weighted_model(scale_pos_weight: float):
    return XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        eval_metric="logloss"
    )


def build_resampled_model():
    return XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        eval_metric="logloss"
    )


def main():
    df = load_data()
    X_train, y_train, X_test, y_test = split_data(df)

    scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()

    experiments = {
        "XGBoost_Weighted": None,
        "XGBoost_RandomOverSampler": RandomOverSampler(random_state=RANDOM_STATE),
        "XGBoost_SMOTE": SMOTE(random_state=RANDOM_STATE),
        "XGBoost_SMOTETomek": SMOTETomek(random_state=RANDOM_STATE),
    }

    for name, sampler in experiments.items():
        print(f"\n===== {name} =====")

        if sampler is None:
            X_resampled, y_resampled = X_train, y_train
            model = build_weighted_model(scale_pos_weight=scale_pos_weight)
        else:
            X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
            model = build_resampled_model()

        print("Training shape:", X_resampled.shape)
        print("Positive rate after resampling:", y_resampled.mean())

        model.fit(X_resampled, y_resampled)

        probs = model.predict_proba(X_test)[:, 1]
        preds = (probs >= THRESHOLD).astype(int)

        pr_auc = average_precision_score(y_test, probs)
        cm = confusion_matrix(y_test, preds)

        print("PR-AUC:", pr_auc)
        print("Threshold:", THRESHOLD)
        print("Confusion Matrix:")
        print(cm)
        print(classification_report(y_test, preds))


if __name__ == "__main__":
    main()
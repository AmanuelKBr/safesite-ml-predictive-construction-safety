import pandas as pd
from sklearn.metrics import average_precision_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import numpy as np

df = pd.read_csv("data/training_data.csv")
df = df.dropna(subset=["high_risk"])

target = "high_risk"

feature_cols = [
    "annual_average_employees",
    "total_hours_worked",
    "total_injuries",
    "total_deaths",
    "total_dafw_cases",
    "total_djtr_cases",
    "total_other_cases",
    "total_dafw_days",
    "total_djtr_days",
    "incident_rate_capped"
]

train_df = df[df["year_filing_for"] <= 2021]
test_df = df[df["year_filing_for"] == 2022]

X_train = train_df[feature_cols].fillna(0)
y_train = train_df[target].astype(int)

X_test = test_df[feature_cols].fillna(0)
y_test = test_df[target].astype(int)

scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)

models = {
    "RandomForest": RandomForestClassifier(
        n_estimators=150,
        class_weight="balanced",
        random_state=42
    ),
    "XGBoost": XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss"
    ),
    "LightGBM": LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=-1,
        scale_pos_weight=scale_pos_weight,
        random_state=42
    )
}

threshold = 0.30  # tuned for higher recall

for name, model in models.items():
    print(f"\n===== {name} =====")
    
    model.fit(X_train, y_train)
    
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= threshold).astype(int)
    
    pr_auc = average_precision_score(y_test, probs)
    
    print("PR-AUC:", pr_auc)
    print("Threshold:", threshold)
    print(classification_report(y_test, preds))
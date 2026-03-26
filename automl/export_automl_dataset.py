# automl/export_automl_dataset.py

import pandas as pd
import numpy as np

DATA_PATH = "data/training_data_severity.csv"
OUTPUT_PATH = "automl/automl_training_dataset.csv"


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["total_deaths"] = pd.to_numeric(df["total_deaths"], errors="coerce").fillna(0).clip(lower=0)
    df["total_hours_worked"] = pd.to_numeric(df["total_hours_worked"], errors="coerce").fillna(0).clip(lower=0)

    df["death_flag"] = (df["total_deaths"] > 0).astype(int)
    df["total_deaths_log"] = np.log1p(df["total_deaths"])

    df["death_severity"] = df["death_flag"] * (1.0 + np.clip(df["total_deaths_log"], 0, 2.0))

    df["death_rate_proxy"] = df["total_deaths"] / (df["total_hours_worked"] + 1.0)
    df["death_rate_proxy_log"] = np.log1p(df["death_rate_proxy"] * 200000.0)

    return df


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

    final_df = df[FEATURE_COLS + [TARGET_COL]].copy()

    final_df.to_csv(OUTPUT_PATH, index=False)

    print("DONE")
    print(f"Saved: {OUTPUT_PATH}")
    print(f"Rows: {len(final_df)}")
    print(f"Columns: {len(final_df.columns)}")
    print("Target: high_risk")


if __name__ == "__main__":
    main()
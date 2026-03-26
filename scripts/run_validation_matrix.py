import json
import joblib
import numpy as np
import pandas as pd

MODEL_PATH = "artifacts/safesite_xgb_weighted_monotonic.joblib"
METADATA_PATH = "artifacts/model_metadata.json"
INPUT_PATH = "data/validation_matrix.csv"
OUTPUT_PATH = "data/validation_results.csv"


def compute_incident_rate(total_injuries, total_hours_worked):
    if pd.isna(total_injuries) or pd.isna(total_hours_worked) or total_hours_worked <= 0:
        return 0.0
    return float((total_injuries * 200000) / total_hours_worked)


def compute_dart_rate(total_dafw_cases, total_djtr_cases, total_hours_worked):
    if pd.isna(total_hours_worked) or total_hours_worked <= 0:
        return 0.0
    dafw = 0.0 if pd.isna(total_dafw_cases) else float(total_dafw_cases)
    djtr = 0.0 if pd.isna(total_djtr_cases) else float(total_djtr_cases)
    return float(((dafw + djtr) * 200000) / total_hours_worked)


def compute_dafw_rate(total_dafw_cases, total_hours_worked):
    if pd.isna(total_dafw_cases) or pd.isna(total_hours_worked) or total_hours_worked <= 0:
        return 0.0
    return float((total_dafw_cases * 200000) / total_hours_worked)


def safe_numeric(value):
    if pd.isna(value):
        return 0.0
    return float(value)


def assign_risk_band(prob, thresholds):
    if prob < thresholds["low"][1]:
        return "Low"
    if prob < thresholds["medium"][1]:
        return "Medium"
    if prob < thresholds["high"][1]:
        return "High"
    return "Critical"


def build_model_features(row, feature_cols):
    raw_tcr_rate = compute_incident_rate(row["total_injuries"], row["total_hours_worked"])
    raw_dart_rate = compute_dart_rate(
        row["total_dafw_cases"],
        row["total_djtr_cases"],
        row["total_hours_worked"]
    )
    raw_dafw_rate = compute_dafw_rate(row["total_dafw_cases"], row["total_hours_worked"])

    total_deaths = safe_numeric(row["total_deaths"])

    # ---- MATCH TRAINING EXACTLY ----
    death_flag = 1.0 if total_deaths > 0 else 0.0
    total_deaths_log = float(np.log1p(total_deaths))

    death_severity = death_flag * (1.0 + min(total_deaths_log, 2.0))

    death_rate_proxy = total_deaths / (safe_numeric(row["total_hours_worked"]) + 1.0)
    death_rate_proxy_log = float(np.log1p(death_rate_proxy * 200000))

    model_features = {
        "annual_average_employees": safe_numeric(row["annual_average_employees"]),
        "total_hours_worked": safe_numeric(row["total_hours_worked"]),
        "death_flag": death_flag,
        "total_deaths_log": total_deaths_log,
        "death_severity": death_severity,
        "death_rate_proxy_log": death_rate_proxy_log,
        "total_dafw_cases": safe_numeric(row["total_dafw_cases"]),
        "total_djtr_cases": safe_numeric(row["total_djtr_cases"]),
        "total_other_cases": safe_numeric(row["total_other_cases"]),
        "total_dafw_days": safe_numeric(row["total_dafw_days"]),
        "total_djtr_days": safe_numeric(row["total_djtr_days"]),
        "tcr_rate_log": float(np.log1p(max(raw_tcr_rate, 0.0))),
        "dart_rate_log": float(np.log1p(max(raw_dart_rate, 0.0))),
        "dafw_rate_log": float(np.log1p(max(raw_dafw_rate, 0.0))),
    }

    input_df = pd.DataFrame([model_features])[feature_cols]

    return input_df, raw_tcr_rate, raw_dart_rate, raw_dafw_rate


def main():
    model_bundle = joblib.load(MODEL_PATH)

    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    model = model_bundle["model"]
    feature_cols = model_bundle["feature_cols"]
    threshold = model_bundle["threshold"]
    thresholds = model_bundle["risk_band_thresholds"]

    df = pd.read_csv(INPUT_PATH)
    results = []

    for _, row in df.iterrows():
        input_df, raw_tcr_rate, raw_dart_rate, raw_dafw_rate = build_model_features(row, feature_cols)

        probability = float(model.predict_proba(input_df)[0][1])
        predicted_class = int(probability >= threshold)
        risk_band = assign_risk_band(probability, thresholds)

        results.append({
            "scenario": row["scenario"],
            "probability": round(probability, 6),
            "predicted_class": predicted_class,
            "risk_band": risk_band,
            "raw_tcr_rate": round(raw_tcr_rate, 4),
            "raw_dart_rate": round(raw_dart_rate, 4),
            "raw_dafw_rate": round(raw_dafw_rate, 4),
            "total_deaths": row["total_deaths"]
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_PATH, index=False)

    print("VALIDATION RESULTS SAVED TO:", OUTPUT_PATH)
    print("\nRISK BAND COUNTS:")
    print(results_df["risk_band"].value_counts(dropna=False))
    print("\nFULL RESULTS:")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
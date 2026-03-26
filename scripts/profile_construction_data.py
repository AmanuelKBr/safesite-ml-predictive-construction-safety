import os
import numpy as np
import pandas as pd
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv

load_dotenv()

connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
container_name = os.getenv("AZURE_STORAGE_CONTAINER")

blob_service_client = BlobServiceClient.from_connection_string(connection_string)
container_client = blob_service_client.get_container_client(container_name)

OUTPUT_DIR = "data"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "training_data_severity.csv")


def load_all_years():
    blobs = container_client.list_blobs()
    dfs = []

    for blob in blobs:
        if blob.name.endswith(".csv"):
            blob_client = container_client.get_blob_client(blob)
            data = blob_client.download_blob().readall()

            df = pd.read_csv(
                pd.io.common.BytesIO(data),
                encoding="latin1",
                low_memory=False
            )
            df["source_file"] = blob.name
            dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


def clean_basic_columns(df):
    df = df.copy()

    if "ï»¿id" in df.columns and "id" in df.columns:
        df = df.drop(columns=["ï»¿id"])
    elif "ï»¿id" in df.columns:
        df = df.rename(columns={"ï»¿id": "id"})

    df["naics_code"] = df["naics_code"].astype(str).str.strip()
    df["year_filing_for"] = pd.to_numeric(df["year_filing_for"], errors="coerce")

    numeric_cols = [
        "annual_average_employees",
        "total_hours_worked",
        "total_injuries",
        "total_deaths",
        "total_dafw_cases",
        "total_djtr_cases",
        "total_other_cases",
        "total_dafw_days",
        "total_djtr_days",
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    return df


def filter_construction(df):
    return df[df["naics_code"].str.startswith("23", na=False)].copy()


def build_establishment_key(df):
    df = df.copy()

    company = df["company_name"].astype(str).str.upper().str.strip()
    est = df["establishment_name"].astype(str).str.upper().str.strip()
    state = df["state"].astype(str).str.upper().str.strip()
    naics = df["naics_code"].astype(str).str.strip()

    df["establishment_key"] = company + "|" + est + "|" + state + "|" + naics
    return df


def create_rate_features(df):
    df = df.copy()

    valid_hours = df["total_hours_worked"].where(df["total_hours_worked"] > 0)

    df["tcr_rate"] = (df["total_injuries"] * 200000) / valid_hours
    df["dart_rate"] = ((df["total_dafw_cases"] + df["total_djtr_cases"]) * 200000) / valid_hours
    df["dafw_rate"] = (df["total_dafw_cases"] * 200000) / valid_hours

    for col in ["tcr_rate", "dart_rate", "dafw_rate"]:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna(0)

    df["tcr_rate_log"] = np.log1p(df["tcr_rate"])
    df["dart_rate_log"] = np.log1p(df["dart_rate"])
    df["dafw_rate_log"] = np.log1p(df["dafw_rate"])

    df["total_deaths"] = df["total_deaths"].clip(lower=0)
    df["death_flag"] = (df["total_deaths"] > 0).astype(int)
    df["total_deaths_log"] = np.log1p(df["total_deaths"])

    return df


def add_yearly_percentiles(df):
    df = df.copy()

    df["tcr_pct"] = df.groupby("year_filing_for")["tcr_rate_log"].rank(pct=True)
    df["dart_pct"] = df.groupby("year_filing_for")["dart_rate_log"].rank(pct=True)
    df["dafw_pct"] = df.groupby("year_filing_for")["dafw_rate_log"].rank(pct=True)
    df["death_pct"] = df.groupby("year_filing_for")["total_deaths_log"].rank(pct=True)

    return df


def create_severity_score(df):
    df = df.copy()

    df["severity_score"] = (
        0.4 * df["tcr_pct"] +
        0.25 * df["dart_pct"] +
        0.15 * df["dafw_pct"] +
        0.2 * df["death_pct"]
    )

    return df


def create_next_year_target(df, top_pct=0.10):
    df = df.copy()

    non_fatal = df[df["death_flag"] == 0]

    thresholds = (
        non_fatal.groupby("year_filing_for")["severity_score"]
        .quantile(1 - top_pct)
        .reset_index()
        .rename(columns={"severity_score": "threshold"})
    )

    next_df = df[[
        "establishment_key",
        "year_filing_for",
        "severity_score",
        "death_flag"
    ]].copy()

    next_df = next_df.merge(thresholds, on="year_filing_for", how="left")

    # FIXED: use int instead of Int64
    next_df["high_risk"] = np.where(
        next_df["death_flag"] == 1,
        1,
        np.where(next_df["severity_score"] >= next_df["threshold"], 1, 0)
    ).astype(int)

    # FORCE dominance
    next_df["severity_score"] = np.where(
        next_df["death_flag"] == 1,
        1.0,
        next_df["severity_score"]
    )

    next_df["feature_year"] = next_df["year_filing_for"] - 1

    next_df = next_df.rename(columns={
        "year_filing_for": "label_year",
        "severity_score": "next_year_severity_score"
    })

    df = df.merge(
        next_df[["establishment_key", "feature_year", "high_risk"]],
        left_on=["establishment_key", "year_filing_for"],
        right_on=["establishment_key", "feature_year"],
        how="left"
    )

    df = df.drop(columns=["feature_year"])
    return df


def create_training_dataset(df):
    return df[df["high_risk"].notna()].copy()


def save(df):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)


if __name__ == "__main__":
    df = load_all_years()
    df = clean_basic_columns(df)
    df = filter_construction(df)
    df = build_establishment_key(df)
    df = create_rate_features(df)
    df = add_yearly_percentiles(df)
    df = create_severity_score(df)
    df = create_next_year_target(df)

    training_df = create_training_dataset(df)
    save(training_df)

    print("DONE:", training_df.shape)
import json
import os
import glob
import joblib
import pandas as pd

model = None

def _find_model_file():
    model_dir = os.getenv("AZUREML_MODEL_DIR", "")
    direct_path = os.path.join(model_dir, "safesite_xgb_weighted_monotonic.joblib")
    if os.path.exists(direct_path):
        return direct_path

    matches = glob.glob(
        os.path.join(model_dir, "**", "safesite_xgb_weighted_monotonic.joblib"),
        recursive=True
    )
    if matches:
        return matches[0]

    raise FileNotFoundError(
        f"Could not find safesite_xgb_weighted_monotonic.joblib under AZUREML_MODEL_DIR={model_dir}"
    )

def init():
    global model
    model_path = _find_model_file()
    loaded = joblib.load(model_path)

    # Handle dict-wrapped model files
    if isinstance(loaded, dict):
        if "model" in loaded:
            model = loaded["model"]
        else:
            raise ValueError(
                f"Model file is a dict but no 'model' key found. Keys present: {list(loaded.keys())}"
            )
    else:
        model = loaded

def run(raw_data):
    try:
        if isinstance(raw_data, str):
            data = json.loads(raw_data)
        else:
            data = raw_data

        if isinstance(data, dict) and "data" in data:
            records = data["data"]
        elif isinstance(data, list):
            records = data
        else:
            records = [data]

        df = pd.DataFrame(records)

        proba = model.predict_proba(df)[:, 1]
        pred = model.predict(df)

        results = []
        for i in range(len(df)):
            p = float(proba[i])

            if p >= 0.85:
                risk_band = "Critical"
            elif p >= 0.60:
                risk_band = "High"
            elif p >= 0.30:
                risk_band = "Medium"
            else:
                risk_band = "Low"

            results.append({
                "probability": round(p, 6),
                "predicted_class": int(pred[i]),
                "risk_band": risk_band
            })

        return {"results": results}

    except Exception as e:
        return {"error": str(e)}

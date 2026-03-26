import json
import math
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb
import altair as alt

st.set_page_config(
    page_title="SafeSite AI",
    page_icon="🦺",
    layout="wide",
    initial_sidebar_state="expanded"
)

MODEL_PATH = "artifacts/safesite_xgb_weighted_monotonic.joblib"
METADATA_PATH = "artifacts/model_metadata.json"
FEATURE_IMPORTANCE_PATH = "artifacts/feature_importance.csv"

BASE_INPUT_KEYS = [
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


@st.cache_resource
def load_artifacts():
    model_bundle = joblib.load(MODEL_PATH)

    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    feature_importance = pd.read_csv(FEATURE_IMPORTANCE_PATH)

    return model_bundle, metadata, feature_importance


def initialize_state():
    if "prediction_generated" not in st.session_state:
        st.session_state["prediction_generated"] = False
    if "reset_counter" not in st.session_state:
        st.session_state["reset_counter"] = 0
    if "last_probability" not in st.session_state:
        st.session_state["last_probability"] = None
    if "last_predicted_class" not in st.session_state:
        st.session_state["last_predicted_class"] = None
    if "last_risk_band" not in st.session_state:
        st.session_state["last_risk_band"] = None
    if "last_raw_tcr_rate" not in st.session_state:
        st.session_state["last_raw_tcr_rate"] = None
    if "last_raw_dart_rate" not in st.session_state:
        st.session_state["last_raw_dart_rate"] = None
    if "last_raw_dafw_rate" not in st.session_state:
        st.session_state["last_raw_dafw_rate"] = None
    if "last_local_explanation_df" not in st.session_state:
        st.session_state["last_local_explanation_df"] = None
    if "last_user_inputs" not in st.session_state:
        st.session_state["last_user_inputs"] = None
    if "last_model_input_df" not in st.session_state:
        st.session_state["last_model_input_df"] = None


def widget_key(name: str) -> str:
    return f"{name}_{st.session_state['reset_counter']}"


def clear_inputs():
    st.session_state["prediction_generated"] = False
    st.session_state["last_probability"] = None
    st.session_state["last_predicted_class"] = None
    st.session_state["last_risk_band"] = None
    st.session_state["last_raw_tcr_rate"] = None
    st.session_state["last_raw_dart_rate"] = None
    st.session_state["last_raw_dafw_rate"] = None
    st.session_state["last_local_explanation_df"] = None
    st.session_state["last_user_inputs"] = None
    st.session_state["last_model_input_df"] = None
    st.session_state["reset_counter"] += 1
    st.rerun()


def inject_custom_css():
    st.markdown(
        """
        <style>
            .block-container {
                max-width: 100% !important;
                padding-top: 1.1rem;
                padding-left: 1.2rem;
                padding-right: 1.2rem;
                padding-bottom: 1.2rem;
            }
            .main-shell {
                background: linear-gradient(180deg, #0b1220 0%, #111827 100%);
                border: 1px solid rgba(255,255,255,0.07);
                border-radius: 22px;
                padding: 22px;
                box-shadow: 0 14px 34px rgba(0,0,0,0.22);
                margin-bottom: 14px;
            }
            .hero-title {
                font-size: 1.95rem;
                font-weight: 800;
                color: #f9fafb;
                margin-bottom: 0.25rem;
            }
            .hero-subtitle {
                color: #cbd5e1;
                font-size: 0.97rem;
                line-height: 1.55;
                margin-bottom: 0.2rem;
            }
            .card {
                background: rgba(255,255,255,0.04);
                border: 1px solid rgba(255,255,255,0.07);
                border-radius: 18px;
                padding: 14px 16px;
                box-shadow: 0 8px 22px rgba(0,0,0,0.14);
                margin-bottom: 12px;
            }
            .section-title {
                color: #f9fafb;
                font-size: 1.03rem;
                font-weight: 700;
                margin-bottom: 0.35rem;
            }
            .muted {
                color: #cbd5e1;
                font-size: 0.92rem;
                line-height: 1.5;
            }
            .result-critical {
                background: linear-gradient(135deg, #7f1d1d, #b91c1c);
                color: white;
                border-radius: 18px;
                padding: 16px 18px;
                border: 1px solid rgba(255,255,255,0.08);
            }
            .result-high {
                background: linear-gradient(135deg, #9a3412, #ea580c);
                color: white;
                border-radius: 18px;
                padding: 16px 18px;
                border: 1px solid rgba(255,255,255,0.08);
            }
            .result-medium {
                background: linear-gradient(135deg, #854d0e, #eab308);
                color: white;
                border-radius: 18px;
                padding: 16px 18px;
                border: 1px solid rgba(255,255,255,0.08);
            }
            .result-low {
                background: linear-gradient(135deg, #166534, #16a34a);
                color: white;
                border-radius: 18px;
                padding: 16px 18px;
                border: 1px solid rgba(255,255,255,0.08);
            }
            .placeholder {
                min-height: 160px;
                display: flex;
                align-items: center;
                justify-content: center;
                text-align: center;
                background: rgba(255,255,255,0.03);
                border: 1px dashed rgba(255,255,255,0.15);
                border-radius: 18px;
                padding: 18px;
            }
            .metric-chip {
                display: inline-block;
                padding: 8px 12px;
                margin-right: 8px;
                margin-top: 8px;
                background: rgba(255,255,255,0.05);
                border: 1px solid rgba(255,255,255,0.08);
                border-radius: 999px;
                color: #e5e7eb;
                font-size: 0.88rem;
            }
            .small-note {
                color: #cbd5e1;
                font-size: 0.86rem;
                line-height: 1.45;
            }
        </style>
        """,
        unsafe_allow_html=True
    )


def safe_float(value: str):
    value = value.strip()
    if value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def safe_numeric_for_model(value):
    return 0.0 if value is None or (isinstance(value, float) and math.isnan(value)) else float(value)


def get_risk_band_thresholds(model_bundle: dict, metadata: dict):
    thresholds = model_bundle.get("risk_band_thresholds", metadata.get("risk_band_thresholds"))

    if thresholds is None:
        return {
            "low": [0.0, 0.30],
            "medium": [0.30, 0.45],
            "high": [0.45, 0.60],
            "critical": [0.60, 1.0]
        }

    # New format: {"low":[0.0,0.3], ...}
    if "low" in thresholds and isinstance(thresholds["low"], list):
        return {
            "low": [float(thresholds["low"][0]), float(thresholds["low"][1])],
            "medium": [float(thresholds["medium"][0]), float(thresholds["medium"][1])],
            "high": [float(thresholds["high"][0]), float(thresholds["high"][1])],
            "critical": [float(thresholds["critical"][0]), float(thresholds["critical"][1])]
        }

    # Old format: {"Low":{"min_inclusive":...,"max_exclusive":...}, ...}
    if "Low" in thresholds and isinstance(thresholds["Low"], dict):
        return {
            "low": [
                float(thresholds["Low"]["min_inclusive"]),
                float(thresholds["Low"]["max_exclusive"])
            ],
            "medium": [
                float(thresholds["Medium"]["min_inclusive"]),
                float(thresholds["Medium"]["max_exclusive"])
            ],
            "high": [
                float(thresholds["High"]["min_inclusive"]),
                float(thresholds["High"]["max_exclusive"])
            ],
            "critical": [
                float(thresholds["Critical"]["min_inclusive"]),
                float(thresholds["Critical"]["max_exclusive"])
            ]
        }

    return {
        "low": [0.0, 0.30],
        "medium": [0.30, 0.45],
        "high": [0.45, 0.60],
        "critical": [0.60, 1.0]
    }


def assign_risk_band(probability: float, risk_band_thresholds: dict) -> str:
    if probability < risk_band_thresholds["low"][1]:
        return "Low"
    if probability < risk_band_thresholds["medium"][1]:
        return "Medium"
    if probability < risk_band_thresholds["high"][1]:
        return "High"
    return "Critical"


def result_css_class(risk_band: str) -> str:
    mapping = {
        "Critical": "result-critical",
        "High": "result-high",
        "Medium": "result-medium",
        "Low": "result-low"
    }
    return mapping.get(risk_band, "result-low")


def compute_incident_rate(total_injuries, total_hours_worked):
    if total_injuries is None or total_hours_worked is None or total_hours_worked <= 0:
        return None
    return (total_injuries * 200000) / total_hours_worked


def compute_dart_rate(total_dafw_cases, total_djtr_cases, total_hours_worked):
    if total_hours_worked is None or total_hours_worked <= 0:
        return None
    dafw = 0 if total_dafw_cases is None else total_dafw_cases
    djtr = 0 if total_djtr_cases is None else total_djtr_cases
    return ((dafw + djtr) * 200000) / total_hours_worked


def compute_dafw_rate(total_dafw_cases, total_hours_worked):
    if total_dafw_cases is None or total_hours_worked is None or total_hours_worked <= 0:
        return None
    return (total_dafw_cases * 200000) / total_hours_worked


def build_model_features(user_inputs: dict, feature_cols: list):
    raw_tcr_rate = compute_incident_rate(
        user_inputs["total_injuries"],
        user_inputs["total_hours_worked"]
    )
    raw_dart_rate = compute_dart_rate(
        user_inputs["total_dafw_cases"],
        user_inputs["total_djtr_cases"],
        user_inputs["total_hours_worked"]
    )
    raw_dafw_rate = compute_dafw_rate(
        user_inputs["total_dafw_cases"],
        user_inputs["total_hours_worked"]
    )

    raw_tcr_rate = 0.0 if raw_tcr_rate is None else float(raw_tcr_rate)
    raw_dart_rate = 0.0 if raw_dart_rate is None else float(raw_dart_rate)
    raw_dafw_rate = 0.0 if raw_dafw_rate is None else float(raw_dafw_rate)

    total_deaths = safe_numeric_for_model(user_inputs["total_deaths"])
    total_hours_worked = safe_numeric_for_model(user_inputs["total_hours_worked"])

    death_flag = 1.0 if total_deaths > 0 else 0.0
    total_deaths_log = float(np.log1p(max(total_deaths, 0.0)))
    death_severity = death_flag * (1.0 + min(total_deaths_log, 2.0))
    death_rate_proxy = total_deaths / (total_hours_worked + 1.0)
    death_rate_proxy_log = float(np.log1p(death_rate_proxy * 200000.0))

    model_features = {
        "annual_average_employees": safe_numeric_for_model(user_inputs["annual_average_employees"]),
        "total_hours_worked": total_hours_worked,
        "death_flag": death_flag,
        "total_deaths_log": total_deaths_log,
        "death_severity": death_severity,
        "death_rate_proxy_log": death_rate_proxy_log,
        "total_dafw_cases": safe_numeric_for_model(user_inputs["total_dafw_cases"]),
        "total_djtr_cases": safe_numeric_for_model(user_inputs["total_djtr_cases"]),
        "total_other_cases": safe_numeric_for_model(user_inputs["total_other_cases"]),
        "total_dafw_days": safe_numeric_for_model(user_inputs["total_dafw_days"]),
        "total_djtr_days": safe_numeric_for_model(user_inputs["total_djtr_days"]),
        "tcr_rate_log": float(np.log1p(max(raw_tcr_rate, 0.0))),
        "dart_rate_log": float(np.log1p(max(raw_dart_rate, 0.0))),
        "dafw_rate_log": float(np.log1p(max(raw_dafw_rate, 0.0))),
    }

    input_df = pd.DataFrame([model_features])

    for col in feature_cols:
        if col not in input_df.columns:
            input_df[col] = np.nan

    input_df = input_df[feature_cols]

    return input_df, raw_tcr_rate, raw_dart_rate, raw_dafw_rate


def format_value(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "Blank → defaulted to 0"
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    return f"{value:,.2f}"


def extract_fitted_xgb_estimator(model):
    try:
        calibrated = model.calibrated_classifiers_[0]
    except Exception:
        return None

    candidates = [
        getattr(calibrated, "estimator", None),
        getattr(calibrated, "base_estimator", None),
    ]

    for candidate in candidates:
        current = candidate
        depth = 0
        while current is not None and depth < 5:
            if hasattr(current, "get_booster"):
                return current
            current = getattr(current, "estimator", None)
            depth += 1

    return None


def get_prediction_explanation(model_bundle, input_df, feature_cols):
    model = model_bundle["model"]
    fitted_model = extract_fitted_xgb_estimator(model)

    if fitted_model is None:
        return pd.DataFrame(columns=["feature", "contribution", "abs_contribution"])

    try:
        dmatrix = xgb.DMatrix(input_df, feature_names=feature_cols)
        contribs = fitted_model.get_booster().predict(dmatrix, pred_contribs=True)[0]
    except Exception:
        return pd.DataFrame(columns=["feature", "contribution", "abs_contribution"])

    contribution_df = pd.DataFrame({
        "feature": feature_cols + ["bias"],
        "contribution": contribs
    })

    feature_contrib_df = contribution_df[contribution_df["feature"] != "bias"].copy()
    feature_contrib_df["abs_contribution"] = feature_contrib_df["contribution"].abs()
    feature_contrib_df = feature_contrib_df.sort_values("abs_contribution", ascending=False)

    return feature_contrib_df


def make_horizontal_bar_chart(df: pd.DataFrame, label_col: str, value_col: str, title: str):
    chart = (
        alt.Chart(df)
        .mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4)
        .encode(
            x=alt.X(f"{value_col}:Q", title=title),
            y=alt.Y(f"{label_col}:N", sort='-x', title=None),
            tooltip=[label_col, alt.Tooltip(value_col, format=".4f")]
        )
        .properties(height=max(260, len(df) * 30))
    )
    return chart


def render_header():
    st.markdown(
        """
        <div class="main-shell">
            <div class="hero-title">SafeSite AI</div>
            <div class="hero-subtitle">
                Predict next-year construction establishment risk from prior-year safety and operating indicators.
            </div>
            <div class="hero-subtitle">
                The app accepts incomplete inputs, generates a risk probability and band, and explains the strongest drivers behind the prediction.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_placeholder():
    st.markdown(
        """
        <div class="placeholder">
            <div>
                <div style="font-size: 1.1rem; font-weight: 700; color: #f9fafb; margin-bottom: 8px;">
                    Prediction output will appear here
                </div>
                <div class="muted">
                    After submission, this panel will show the risk band, probability, local drivers, global drivers, and input summary.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_glossary_card():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Rate Glossary</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="small-note">
            <strong>TCR Rate</strong><br>
            Total Case Rate = (total injuries × 200,000) ÷ total hours worked
            <br><br>
            <strong>DART Rate</strong><br>
            Days Away, Restricted, or Transferred = ((DAFW cases + DJTR cases) × 200,000) ÷ total hours worked
            <br><br>
            <strong>DAFW Rate</strong><br>
            Days Away From Work Rate = (DAFW cases × 200,000) ÷ total hours worked
            <br><br>
            <strong>Death Severity</strong><br>
            Derived fatality feature used by the model to materially elevate fatal cases above otherwise similar non-fatal cases.
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown('</div>', unsafe_allow_html=True)


def render_about_card(metadata, risk_band_thresholds):
    low_max = risk_band_thresholds["low"][1]
    med_min = risk_band_thresholds["medium"][0]
    med_max = risk_band_thresholds["medium"][1]
    high_min = risk_band_thresholds["high"][0]
    high_max = risk_band_thresholds["high"][1]
    crit_min = risk_band_thresholds["critical"][0]

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">About the Product</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="small-note">
            <strong>Source:</strong> {metadata.get("data_source", "OSHA ITA construction training dataset")}<br><br>
            <strong>Target:</strong> {metadata.get("target_definition", "Next-year high-risk classification from severity-based target engineering")}<br><br>
            <strong>Prediction horizon:</strong> Next year<br><br>
            <strong>Input behavior:</strong> Blank fields are accepted and defaulted to 0 in the current UI inference pipeline.<br><br>
            <strong>Risk bands:</strong><br>
            • Critical: probability ≥ {crit_min:.2f}<br>
            • High: {high_min:.2f} to &lt; {high_max:.2f}<br>
            • Medium: {med_min:.2f} to &lt; {med_max:.2f}<br>
            • Low: &lt; {low_max:.2f}
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown('</div>', unsafe_allow_html=True)


def main():
    initialize_state()
    inject_custom_css()
    model_bundle, metadata, feature_importance = load_artifacts()
    render_header()

    model = model_bundle["model"]
    feature_cols = model_bundle["feature_cols"]
    threshold = float(model_bundle.get("threshold", metadata.get("threshold", 0.30)))
    risk_band_thresholds = get_risk_band_thresholds(model_bundle, metadata)

    left, right = st.columns([0.92, 1.48], gap="large")

    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Input Panel</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="muted">Enter prior-year values you know. Blank fields are allowed. In the current UI pipeline, blanks default to 0 for prediction.</div>',
            unsafe_allow_html=True
        )
        st.caption("Prediction horizon: next year")

        with st.form("prediction_form"):
            c1, c2 = st.columns(2)

            with c1:
                annual_average_employees = st.text_input("Annual Average Employees", key=widget_key("annual_average_employees"), placeholder="125")
                total_hours_worked = st.text_input("Total Hours Worked", key=widget_key("total_hours_worked"), placeholder="250000")
                total_injuries = st.text_input("Total Injuries", key=widget_key("total_injuries"), placeholder="12")
                total_deaths = st.text_input("Total Deaths", key=widget_key("total_deaths"), placeholder="0")
                total_dafw_cases = st.text_input("DAFW Cases", key=widget_key("total_dafw_cases"), placeholder="4")

            with c2:
                total_djtr_cases = st.text_input("DJTR Cases", key=widget_key("total_djtr_cases"), placeholder="2")
                total_other_cases = st.text_input("Other Cases", key=widget_key("total_other_cases"), placeholder="6")
                total_dafw_days = st.text_input("DAFW Days", key=widget_key("total_dafw_days"), placeholder="45")
                total_djtr_days = st.text_input("DJTR Days", key=widget_key("total_djtr_days"), placeholder="18")

            submitted = st.form_submit_button("Predict Next-Year Risk", use_container_width=True)

        b1, b2 = st.columns([1, 1])
        with b1:
            if st.button("Clear Inputs", use_container_width=True):
                clear_inputs()
        with b2:
            st.empty()

        st.markdown('</div>', unsafe_allow_html=True)

        with st.expander("Model and project details", expanded=True):
            st.write(f"Algorithm: {metadata.get('algorithm', 'Calibrated XGBoost')}")
            if "pr_auc" in metadata:
                st.write(f"PR-AUC: {metadata['pr_auc']:.3f}")
            if "positive_recall" in metadata:
                st.write(f"Positive Recall: {metadata['positive_recall']:.2f}")
            if "positive_precision" in metadata:
                st.write(f"Positive Precision: {metadata['positive_precision']:.2f}")
            if "positive_f1" in metadata:
                st.write(f"Positive F1: {metadata['positive_f1']:.2f}")
            st.write(f"Threshold: {threshold:.2f}")
            st.write(f"Data Source: {metadata.get('data_source', 'OSHA ITA construction dataset')}")
            st.write(f"Target: {metadata.get('target_definition', 'Next-year high-risk classification')}")
            if "feature_design_note" in metadata:
                st.write(f"Feature design: {metadata['feature_design_note']}")
            else:
                st.write("Feature design: includes injury-rate signals plus death-focused features (death_flag, total_deaths_log, death_severity, death_rate_proxy_log).")

    with right:
        if submitted:
            st.session_state["prediction_generated"] = True

            user_inputs = {
                "annual_average_employees": safe_float(annual_average_employees),
                "total_hours_worked": safe_float(total_hours_worked),
                "total_injuries": safe_float(total_injuries),
                "total_deaths": safe_float(total_deaths),
                "total_dafw_cases": safe_float(total_dafw_cases),
                "total_djtr_cases": safe_float(total_djtr_cases),
                "total_other_cases": safe_float(total_other_cases),
                "total_dafw_days": safe_float(total_dafw_days),
                "total_djtr_days": safe_float(total_djtr_days),
            }

            input_df, raw_tcr_rate, raw_dart_rate, raw_dafw_rate = build_model_features(user_inputs, feature_cols)

            probability = float(model.predict_proba(input_df)[0][1])
            predicted_class = int(probability >= threshold)
            risk_band = assign_risk_band(probability, risk_band_thresholds)

            local_explanation_df = get_prediction_explanation(
                model_bundle=model_bundle,
                input_df=input_df,
                feature_cols=feature_cols
            )

            st.session_state["last_probability"] = probability
            st.session_state["last_predicted_class"] = predicted_class
            st.session_state["last_risk_band"] = risk_band
            st.session_state["last_raw_tcr_rate"] = raw_tcr_rate
            st.session_state["last_raw_dart_rate"] = raw_dart_rate
            st.session_state["last_raw_dafw_rate"] = raw_dafw_rate
            st.session_state["last_local_explanation_df"] = local_explanation_df
            st.session_state["last_user_inputs"] = user_inputs
            st.session_state["last_model_input_df"] = input_df.copy()

        if not st.session_state["prediction_generated"]:
            render_placeholder()

            tab1, tab2, tab3 = st.tabs(["Global Drivers", "Input Guide", "About"])

            with tab1:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown('<div class="section-title">Global Feature Importance</div>', unsafe_allow_html=True)
                top_global = feature_importance.head(10).copy()
                st.altair_chart(make_horizontal_bar_chart(top_global, "feature", "importance", "Importance"), use_container_width=True)
                st.dataframe(top_global, use_container_width=False, hide_index=True, width=520)
                st.markdown('</div>', unsafe_allow_html=True)

            with tab2:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown('<div class="section-title">Input Guide</div>', unsafe_allow_html=True)
                guide_df = pd.DataFrame({
                    "field": [
                        "Annual Average Employees", "Total Hours Worked", "Total Injuries",
                        "Total Deaths", "DAFW Cases", "DJTR Cases", "Other Cases",
                        "DAFW Days", "DJTR Days",
                    ],
                    "description": [
                        "Average headcount for the prior year",
                        "Total prior-year hours worked",
                        "Prior-year total injury count",
                        "Prior-year death count",
                        "Days away from work cases",
                        "Job transfer or restriction cases",
                        "Other OSHA-recordable cases",
                        "Total days away from work",
                        "Total restricted or transferred days",
                    ]
                })
                st.dataframe(guide_df, use_container_width=True, hide_index=True)
                render_glossary_card()

            with tab3:
                render_about_card(metadata, risk_band_thresholds)

        else:
            probability = st.session_state["last_probability"]
            predicted_class = st.session_state["last_predicted_class"]
            risk_band = st.session_state["last_risk_band"]
            raw_tcr_rate = st.session_state["last_raw_tcr_rate"]
            raw_dart_rate = st.session_state["last_raw_dart_rate"]
            raw_dafw_rate = st.session_state["last_raw_dafw_rate"]
            local_explanation_df = st.session_state["last_local_explanation_df"]
            user_inputs = st.session_state["last_user_inputs"]
            model_input_df = st.session_state["last_model_input_df"]

            st.markdown(
                f"""
                <div class="{result_css_class(risk_band)}">
                    <div style="font-size: 0.95rem; opacity: 0.95;">Predicted Next-Year Risk Band</div>
                    <div style="font-size: 2rem; font-weight: 800; margin-top: 6px;">{risk_band}</div>
                    <div style="margin-top: 8px; font-size: 1rem;">
                        Probability of High Risk: <strong>{probability:.1%}</strong>
                    </div>
                    <div style="margin-top: 6px; font-size: 0.95rem;">
                        Predicted High-Risk Class: <strong>{predicted_class}</strong>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown(
                f"""
                <span class="metric-chip">TCR Rate: {format_value(raw_tcr_rate)}</span>
                <span class="metric-chip">DART Rate: {format_value(raw_dart_rate)}</span>
                <span class="metric-chip">DAFW Rate: {format_value(raw_dafw_rate)}</span>
                <span class="metric-chip">Risk Band: {risk_band}</span>
                """,
                unsafe_allow_html=True
            )

            tab1, tab2, tab3 = st.tabs(["Prediction Drivers", "Global Drivers", "Input Summary"])

            with tab1:
                content_left, content_right = st.columns([1.15, 0.85], gap="large")

                with content_left:
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.markdown('<div class="section-title">Per-Prediction Risk Drivers</div>', unsafe_allow_html=True)

                    if local_explanation_df is None or local_explanation_df.empty:
                        st.info("Prediction contribution details are unavailable for the calibrated wrapper, but the model prediction is valid.")
                    else:
                        top_local = local_explanation_df.head(10)[["feature", "contribution"]].copy()
                        st.altair_chart(
                            make_horizontal_bar_chart(top_local, "feature", "contribution", "Contribution"),
                            use_container_width=True
                        )
                        st.dataframe(top_local, use_container_width=False, hide_index=True, width=430)
                    st.markdown('</div>', unsafe_allow_html=True)

                with content_right:
                    render_about_card(metadata, risk_band_thresholds)
                    render_glossary_card()

            with tab2:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown('<div class="section-title">Global Feature Importance</div>', unsafe_allow_html=True)
                top_global = feature_importance.head(10).copy()
                st.altair_chart(make_horizontal_bar_chart(top_global, "feature", "importance", "Importance"), use_container_width=True)
                st.dataframe(top_global, use_container_width=False, hide_index=True, width=520)
                st.markdown('</div>', unsafe_allow_html=True)

            with tab3:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown('<div class="section-title">Submitted Input Summary</div>', unsafe_allow_html=True)

                raw_input_summary = pd.DataFrame({
                    "field": [
                        "annual_average_employees", "total_hours_worked", "total_injuries",
                        "total_deaths", "total_dafw_cases", "total_djtr_cases",
                        "total_other_cases", "total_dafw_days", "total_djtr_days",
                    ],
                    "entered_value": [
                        user_inputs.get("annual_average_employees"),
                        user_inputs.get("total_hours_worked"),
                        user_inputs.get("total_injuries"),
                        user_inputs.get("total_deaths"),
                        user_inputs.get("total_dafw_cases"),
                        user_inputs.get("total_djtr_cases"),
                        user_inputs.get("total_other_cases"),
                        user_inputs.get("total_dafw_days"),
                        user_inputs.get("total_djtr_days"),
                    ],
                })
                raw_input_summary["display_value"] = raw_input_summary["entered_value"].apply(format_value)

                st.markdown("**Raw entered inputs**")
                st.dataframe(raw_input_summary[["field", "display_value"]], use_container_width=True, hide_index=True)

                st.markdown("**Derived model inputs**")
                model_input_summary = pd.DataFrame({
                    "feature": model_input_df.columns.tolist(),
                    "value": model_input_df.iloc[0].tolist()
                })
                model_input_summary["display_value"] = model_input_summary["value"].apply(format_value)
                st.dataframe(model_input_summary[["feature", "display_value"]], use_container_width=True, hide_index=True)

                st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
# SafeSite ML — Predictive Construction Safety System

An end-to-end machine learning project that predicts next-year construction safety risk from OSHA-style historical safety data and demonstrates multiple enterprise deployment patterns.

## Project Summary

SafeSite ML was built to predict whether a construction establishment is likely to fall into a high-risk safety category in the following year. The system was designed not only for overall predictive performance, but also for realistic sensitivity to severe safety events such as fatalities.

This project demonstrates:

- machine learning for rare and safety-critical outcomes
- feature engineering for construction safety data
- calibrated classification with risk bands
- Streamlit app development
- Docker containerization
- Azure Container Apps deployment
- Azure ML managed online endpoint deployment
- Azure ML AutoML benchmarking for model comparison

## Business Problem

In construction safety, overall accuracy alone is not enough. A model that performs well on common low-risk patterns but underreacts to fatalities is not operationally trustworthy.

The core business requirement for this project was:

- predict next-year high-risk safety outcome
- produce probability and interpretable risk bands
- ensure fatality-heavy cases receive appropriately elevated risk
- compare alternative modeling approaches without losing the validated production baseline

## Data Scope

- Domain: OSHA ITA-style construction safety data
- Industry scope: NAICS 23 construction sector
- Years covered: 2016–2023
- Modeling dataset size: 116,415 rows after final feature engineering and filtering
- Final modeling dataset used for AutoML benchmarking: 14 features + 1 target (`high_risk`)

## Final Feature Set

The final production model used these features:

- annual_average_employees
- total_hours_worked
- death_flag
- total_deaths_log
- death_severity
- death_rate_proxy_log
- total_dafw_cases
- total_djtr_cases
- total_other_cases
- total_dafw_days
- total_djtr_days
- tcr_rate_log
- dart_rate_log
- dafw_rate_log

Target:

- `high_risk`

## Why Weighted Imbalance Handling Was Chosen Over Oversampling

Class imbalance handling was a critical modeling decision.

Weighted learning was selected instead of oversampling because:

- the dataset was already large enough
- oversampling rare fatal cases would duplicate patterns and increase overfitting risk
- weighting preserved the real data distribution
- weighting is easier to justify for tree-based models in structured tabular data
- the objective was to force the model to pay more attention to fatal cases without fabricating synthetic records

In the final successful training setup, fatal cases in training were given substantially higher sample weight, which materially improved fatality sensitivity.

## Modeling Journey

Early versions of the model underreacted to fatalities. Even death-heavy scenarios sometimes remained in Low or Medium risk bands, which was unacceptable for the business objective.

The final successful path included:

1. stronger fatality-focused feature engineering  
   - `death_flag`
   - `total_deaths_log`
   - `death_severity`
   - `death_rate_proxy_log`

2. target engineering alignment  
   Fatal cases were explicitly treated as high-severity patterns in the target logic.

3. weighted training  
   Fatal training cases received higher weight so the learner paid more attention to those cases.

4. calibration and validation alignment  
   The calibrated model and validation pipeline were corrected so deployment behavior matched training intent.

This combination produced the final realistic fatality-sensitive baseline.

## Final Production Model

The selected production model is:

- **Weighted + calibrated XGBoost classifier**

Why it was selected:

- handled nonlinear structured relationships well
- supported the rare-event/fatality-focused objective better than simpler baselines
- produced the strongest practical balance of predictive power and safety realism
- worked well with explicit fatality feature engineering and weighted learning
- remained stable for deployment in both app and endpoint formats

## AutoML Benchmarking

Azure ML AutoML was used as a separate benchmark track to test whether another model family could outperform the selected baseline without replacing it prematurely.

### AutoML best model

- **VotingEnsemble**

### AutoML result highlights

- Accuracy: **0.898**
- F1 weighted: **0.858**
- AUC weighted: **0.792**
- Balanced accuracy: **0.529**
- Recall macro: **0.529**

### Why AutoML was not selected for production

Although the AutoML VotingEnsemble had good overall metrics, its balanced accuracy and macro recall were weak. That indicated weaker minority/severe-class behavior, which was a critical business requirement in this project.

For that reason, the AutoML model was retained as a benchmark artifact, but the weighted XGBoost model remained the production choice.

## Model Comparison Summary

| Model / Approach | Purpose | Strength | Limitation | Decision |
|---|---|---|---|---|
| Initial tree / baseline approaches | Early experimentation | Established starting point | Underreacted to fatalities | Rejected |
| Oversampling-based XGBoost | Imbalance experiment | Improved exposure to rare class | Higher overfitting risk on rare fatal patterns | Rejected |
| Weighted XGBoost + calibration | Production candidate | Best fatal sensitivity and deployable stability | Required careful calibration and validation alignment | **Selected** |
| Azure ML AutoML VotingEnsemble | Benchmark model | Strong overall metrics | Weak balanced accuracy and macro recall for minority/severe behavior | Benchmark only |

## Risk Output Design

The final system produces:

- predicted probability
- binary predicted class
- risk band:
  - Low
  - Medium
  - High
  - Critical

Risk bands are derived from probability thresholds, with the final thresholding logic saved as metadata for reuse in the app and endpoint.

## Solution Architecture

This project showcases one production baseline model delivered through multiple consumption paths.

### 1) Local development path

- Python training scripts
- Streamlit app for local testing
- local artifact-based inference

### 2) Cloud application path

- Dockerized Streamlit app
- image pushed to Azure Container Registry
- deployed to Azure Container Apps
- public web app for recruiter-friendly demonstration

### 3) Enterprise ML serving path

- registered model in Azure ML
- custom scoring script (`score.py`)
- managed online endpoint for REST inference
- endpoint tested successfully with sample request payload

### 4) Benchmarking path

- Azure ML AutoML classification job
- benchmark model registered separately
- results compared without disturbing the production baseline

## Repository Structure

```text
.
├── app.py
├── Dockerfile
├── requirements.txt
├── requirements-docker.txt
├── score.py
├── conda.yaml
├── endpoint.yml
├── deployment.yaml
├── environment.yaml
├── sample-request.json
├── artifacts/
│   ├── feature_importance.csv
│   └── model_metadata.json
├── automl/
│   └── export_automl_dataset.py
└── scripts/
    ├── compare_imbalance_methods.py
    ├── data_loader.py
    ├── profile_construction_data.py
    ├── run_validation_matrix.py
    ├── train_final_model.py
    └── train_model.py
```

## Key Files

| File | Role |
|---|---|
| `scripts/train_final_model.py` | Final weighted XGBoost training pipeline |
| `scripts/run_validation_matrix.py` | Validation scenarios for behavior testing |
| `app.py` | Streamlit user interface |
| `Dockerfile` | Container image definition for app deployment |
| `score.py` | Azure ML inference entry point |
| `sample-request.json` | Example REST scoring payload |
| `artifacts/model_metadata.json` | Saved feature and threshold metadata |
| `artifacts/feature_importance.csv` | Feature importance output from final model |
| `automl/export_automl_dataset.py` | Recreates final modeling dataset for AutoML benchmarking |

## How to Run Locally

1. Create and activate a virtual environment
2. Install dependencies
3. Run the Streamlit app

Example:

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Azure Deployment Summary

### Azure Container Apps
Used for the recruiter-facing web application deployment.

### Azure ML Managed Online Endpoint
Used for production-style model serving through REST API.

### Azure ML AutoML
Used to benchmark alternative models while preserving the validated weighted XGBoost baseline.

## Portfolio Value

This project demonstrates:

- machine learning on safety-critical tabular data
- rare-event handling with business-aware evaluation
- explainable model selection decisions
- cloud deployment across app and endpoint patterns
- Azure ML operational maturity beyond notebook-only work
- ability to compare models using both metrics and domain behavior

## Future Enhancements

- CI/CD for automated retraining and deployment
- model monitoring and drift checks
- richer explainability layer for prediction factors
- automated endpoint and app health validation
- optional Power BI reporting layer for risk trend consumption

## Author

**Amanuel Birri**

GitHub: `https://github.com/AmanuelKBr/safesite-ml-predictive-construction-safety`
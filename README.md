# SafeSite ML — Predictive Construction Safety System

## Live Demo

Access the deployed Streamlit application here:

🔗 https://ca-safesite-ml-ui-dev-01.orangepond-e5ed1e5d.westus2.azurecontainerapps.io/

This is the primary demonstration interface for the project. The app allows users to enter construction safety inputs and receive:

- predicted probability of next-year high-risk outcome
- binary classification
- risk band:
  - Low
  - Medium
  - High
  - Critical

## Project Summary

SafeSite ML is an end-to-end machine learning project that predicts next-year construction safety risk from OSHA-style historical data and demonstrates multiple enterprise deployment patterns.

The project was built not just for overall predictive performance, but for realistic sensitivity to severe safety events such as fatalities. It combines model development, validation, containerization, cloud deployment, REST endpoint serving, and AutoML benchmarking in one portfolio-ready solution.

This project demonstrates:

- machine learning for rare and safety-critical outcomes
- feature engineering for construction safety data
- calibrated classification with interpretable risk bands
- Streamlit application development
- Docker containerization
- Azure Container Apps deployment
- Azure ML managed online endpoint deployment
- Azure ML AutoML benchmarking for model comparison

## Business Problem

In construction safety, overall accuracy alone is not enough. A model that performs well on common low-risk patterns but underreacts to fatalities is not operationally trustworthy.

The central business requirement for this project was to:

- predict next-year high-risk safety outcome
- produce probability and risk-band outputs
- ensure fatality-heavy cases receive appropriately elevated risk
- compare alternative modeling approaches without losing the validated production baseline

## Data Scope

- Domain: OSHA ITA-style construction safety data
- Industry scope: NAICS 23 construction sector
- Years covered: 2016–2023
- Final modeling dataset size: 116,415 rows
- Final modeling dataset shape for benchmarking: 14 features + 1 target (`high_risk`)

## Final Feature Set

The final production model uses the following features:

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

Class imbalance handling was one of the most important modeling decisions in the project.

Weighted learning was selected instead of oversampling because:

- the dataset was already large enough
- oversampling rare fatal cases would duplicate patterns and increase overfitting risk
- weighting preserved the original data distribution
- weighting is cleaner and easier to justify for tree-based models on structured tabular data
- the goal was to force the learner to pay more attention to fatal cases without fabricating synthetic examples

In the final successful training setup, fatal cases in training were assigned substantially higher sample weight. This materially improved fatality sensitivity and was one of the key reasons the final model behaved more realistically on death-heavy scenarios.

## Modeling Journey

Early versions of the model underreacted to fatalities. Even death-heavy scenarios could remain in Low or Medium risk bands, which was unacceptable for the business objective.

The final successful path included:

1. stronger fatality-focused feature engineering
   - `death_flag`
   - `total_deaths_log`
   - `death_severity`
   - `death_rate_proxy_log`

2. target engineering alignment  
   Fatal cases were explicitly treated as high-severity patterns in the target logic.

3. weighted training  
   Fatal training cases received higher sample weight so the learner paid more attention to those cases.

4. calibration and validation alignment  
   The calibrated model and validation pipeline were corrected so deployment behavior matched training intent.

This combination produced the final realistic fatality-sensitive baseline.

## Final Production Model

The selected production model is:

- **Weighted + calibrated XGBoost classifier**

Why it was selected:

- it handled nonlinear structured relationships well
- it supported the rare-event and fatality-focused objective better than simpler baselines
- it produced the strongest practical balance of predictive power and safety realism
- it worked well with explicit fatality feature engineering and weighted learning
- it remained stable for deployment in both app and endpoint formats

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

Although the AutoML VotingEnsemble produced good overall metrics, its balanced accuracy and macro recall were weak. That pointed to weaker minority and severe-class behavior, which was a critical business requirement in this project.

For that reason, the AutoML model was retained as a benchmark artifact, while the weighted XGBoost model remained the production choice.

## Model Comparison Summary

| Model / Approach | Purpose | Strength | Limitation | Decision |
|---|---|---|---|---|
| Initial baseline approaches | Early experimentation | Established starting point | Underreacted to fatalities | Rejected |
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

Risk bands are derived from probability thresholds, with the final thresholding logic saved as metadata for reuse in both the app and endpoint.

## Solution Architecture

This project showcases one validated production baseline model delivered through multiple consumption paths.

### 1) Local development path

- Python training scripts
- Streamlit app for local testing
- local artifact-based inference

### 2) Cloud application path

- Dockerized Streamlit app
- image pushed to Azure Container Registry
- deployed to Azure Container Apps
- public web application for recruiter-friendly demonstration

### 3) Enterprise ML serving path

- registered model in Azure ML
- custom scoring script (`score.py`)
- managed online endpoint for REST inference
- endpoint tested successfully with sample request payload

### 4) Benchmarking path

- Azure ML AutoML classification job
- benchmark model registered separately
- results compared without disturbing the production baseline

## Deployment Paths Included in This Repository

This repository documents and supports all major versions built during the project:

1. **Local Streamlit app**  
   Local execution using the trained weighted XGBoost artifact.

2. **Dockerized Streamlit app**  
   Containerized version of the application for consistent packaging and deployment.

3. **Azure Container Apps deployment**  
   Public cloud-hosted Streamlit interface for live demonstration.

4. **Azure ML managed online endpoint**  
   Production-style REST inference endpoint for enterprise model serving.

5. **Azure ML AutoML benchmark track**  
   Separate benchmarking workflow used to compare alternative models without replacing the selected production baseline.

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

## Azure ML Endpoint (REST API Example)

This project also exposes the model through an Azure ML managed online endpoint.

Example request payload:

```json
{
  "data": [
    {
      "annual_average_employees": 100,
      "total_hours_worked": 200000,
      "death_flag": 1,
      "total_deaths_log": 1.79,
      "death_severity": 2.5,
      "death_rate_proxy_log": 2.1,
      "total_dafw_cases": 10,
      "total_djtr_cases": 5,
      "total_other_cases": 2,
      "total_dafw_days": 100,
      "total_djtr_days": 50,
      "tcr_rate_log": 2.0,
      "dart_rate_log": 1.8,
      "dafw_rate_log": 1.5
    }
  ]
}
```

This endpoint demonstrates production-style model serving separate from the web UI layer.

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
Used for the recruiter-facing live web application deployment.

### Azure ML Managed Online Endpoint
Used for production-style model serving through REST API.

### Azure ML AutoML
Used to benchmark alternative models while preserving the validated weighted XGBoost baseline.

## Portfolio Value

This project demonstrates:

- machine learning on safety-critical tabular data
- rare-event handling with business-aware evaluation
- explainable model selection decisions
- cloud deployment across both app and endpoint patterns
- Azure ML operational maturity beyond notebook-only workflows
- model comparison using both metrics and domain behavior

## Future Enhancements

- CI/CD for automated retraining and deployment
- model monitoring and drift checks
- richer explainability layer for prediction factors
- automated endpoint and app health validation
- optional Power BI reporting layer for risk trend consumption

## Author

**Amanuel Birri**

GitHub: `https://github.com/AmanuelKBr/safesite-ml-predictive-construction-safety`
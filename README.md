# Municipal Energy Demand Forecasting Pipeline

An empirical machine learning framework for multivariate time-series forecasting of ward-level electrical grid consumption variances, supporting urban sustainability and municipal conservation planning.

## Architecture & Features
- **Temporal Modeling:** Generates multi-step lag features and rolling averages from weather telemetry and occupancy metrics.
- **Predictive Engine:** Utilizes gradient boosting (`XGBoost`) with chronological time-series cross-validation.
- **Evidence-Based Policy Synthesis:** Generates automated demand-response insights formatted for public utility stakeholders.

## Directory Structure
```text
├── data/               # Microclimate telemetry and ward consumption records
├── src/models/         # Lag generators and XGBoost model definitions
├── reports/            # Municipal policy briefs and empirical summaries
└── requirements.txt    # Project dependencies

```

## Running the Forecaster
```bash
pip install -r requirements.txt
python src/models/xgboost_forecaster.py
```

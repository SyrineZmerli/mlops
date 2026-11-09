# MLOps Project

A complete MLOps pipeline for churn prediction using MLflow, FastAPI, and scikit-learn.

## Project Structure

- `train_mlflow.py` — Train and log models with MLflow
- `register_model.py` — Register the best model to the MLflow Model Registry
- `load_champion.py` — Load the champion model alias
- `app.py` — FastAPI serving the champion model
- `test_api.py` — API tests
- `churn_model.ipynb` — Exploratory notebook

## Branches

| Branch | Description |
|--------|-------------|
| `master` | Stable base |
| `develop` | Integration branch |
| `mlflow` | MLflow tracking & experiments |
| `TP4/resistry-model-fastAPI` | Model registry + FastAPI serving |

## Stack

- Python, scikit-learn
- MLflow (tracking + registry)
- FastAPI + Pydantic
- SQLite (MLflow backend store)

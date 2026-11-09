import mlflow
from mlflow import MlflowClient

# Configuration constants
TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "churn-prediction"
MODEL_NAME = "churn-model"
ALIAS_NAME = "champion"
MODEL_ARTIFACT_PATH = "model"

# Setup MLflow connection
mlflow.set_tracking_uri(TRACKING_URI)
client = MlflowClient()

# 1. Retrieve the best run based on F1 score (using >= 0 instead of IS NOT NULL)
runs = mlflow.search_runs(
	experiment_names=[EXPERIMENT_NAME],
	filter_string="metrics.f1 >= 0",
	order_by=["metrics.f1 DESC"],
	max_results=1
)

if runs.empty:
	raise ValueError(f"No valid runs with logged 'metrics.f1' found in experiment '{EXPERIMENT_NAME}'.")

best_run = runs.iloc[0]
run_id = best_run["run_id"]
best_f1 = best_run.get("metrics.f1", 0.0)
run_name = best_run.get("tags.mlflow.runName", "N/A")

print(f"Best run: {run_name} | ID: {run_id[:8]}... | F1: {best_f1:.4f}")

# 2. Register the best model to the Registry
model_uri = f"runs:/{run_id}/{MODEL_ARTIFACT_PATH}"
registered_model = mlflow.register_model(model_uri, MODEL_NAME)

print(f"Model registered: {MODEL_NAME} | Version: {registered_model.version}")

# 3. Assign the 'champion' alias
client.set_registered_model_alias(MODEL_NAME, ALIAS_NAME, registered_model.version)
print(f"Alias '{ALIAS_NAME}' assigned -> Version {registered_model.version}")
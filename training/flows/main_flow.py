import mlflow
from mlflow.tracking import MlflowClient
from prefect import flow, task
import pandas as pd
from sklearn.metrics import recall_score
import os
from .model_training import prepare_data, train_lr, train_xgb, train_rf
experiment_name = "predictive-maintenance-prediction"
bucket = "s3://predictive-maintenance-artifacts-victor-obi/mlflow"
mlflow_tracking_uri = "sqlite:///mlflow.db"

@task
def setup_mlfow():
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if not experiment:
        mlflow.create_experiment(name=experiment_name, artifact_location=bucket)
    mlflow.set_experiment(experiment_name)
    client = MlflowClient(tracking_uri=mlflow_tracking_uri)
    return client

@task
def load_data():
    X_train, y_train = prepare_data("data/partition/train.parquet")
    X_test, y_test = prepare_data("data/partition/test.parquet")
    return X_train, X_test, y_train, y_test

@task
def run_all_experiments(X_train, X_test, y_train, y_test):
    setup_mlfow()
    xgb_run_id, xgb_recall = train_xgb(X_train, y_train, X_test, y_test)
    lr_run_id, lr_recall = train_lr(X_train, y_train, X_test, y_test)
    rf_run_id, rf_recall = train_rf(X_train, y_train, X_test, y_test)
    results = [
        ("XGBoost", xgb_run_id, xgb_recall),
        ("RandomForest", rf_run_id, rf_recall),
        ("LogReg", lr_run_id, lr_recall)
    ]
    results.sort(key=lambda x: x[2], reverse=True)
    winner_name, winner_id, winner_score = results[0]

    print(f"Best model: {winner_name} with Recall: {winner_score:.4f}")
    return winner_id, winner_score

@task
def promote_best_model(winner_id, winner_score):
    client = setup_mlfow()
    try:
        prod_models = client.get_latest_versions(name=experiment_name, stages=["Production"])
        current_prod = prod_models[0]
        prod_run = client.get_run(current_prod.run_id)
        prod_recall = prod_run.data.metrics['recall']
    except:
        prod_recall = 0.0

    if winner_score > prod_recall:
        model_url = f"runs:/{winner_id}/model"
        mv = mlflow.register_model(model_url, name=experiment_name)

        client.transition_model_version_stage(
            name=experiment_name,
            version=mv.version,
            stage='Production',
            archive_existing_versions=True
        )

@flow(name="Predictive Maintenance Training Flow")
def main_flow():
    setup_mlfow()
    X_train, X_test, y_train, y_test = load_data()
    winner_id, winner_score = run_all_experiments(X_train, X_test, y_train, y_test)
    promote_best_model(winner_id, winner_score)

if __name__ == "__main__":
    main_flow()
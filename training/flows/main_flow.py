import mlflow
from mlflow.tracking import MlflowClient
from prefect import flow, task
import dagshub
from dotenv import load_dotenv
from model_training import prepare_data, train_lr, train_xgb, train_rf
import os 
import subprocess
import sys


load_dotenv()
experiment_name = "Predictive-Maintenance-Experiment-v2"
bucket = "s3://predictive-maintenance-artifacts-victor-obi/mlflow"


@task(log_prints=True)
def pull_dvc_data():
    print("Starting DVC Pull from S3...")
    
    result = subprocess.run(["dvc", "pull", "-v"], capture_output=True, text=True)
    
    if result.returncode == 0:
        print("Data pulled successfully!")
        print(result.stdout)
    else:
        print("DVC Pull Failed!")
        print(result.stderr)
        raise Exception("DVC Pull failed")
    

def setup_mlflow():
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    
    if not tracking_uri:
        print("Warning: MLFLOW_TRACKING_URI not found. Saving locally.")
    else:
        mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if not experiment:
        mlflow.create_experiment(name=experiment_name, artifact_location=bucket)
    mlflow.set_experiment(experiment_name)
    return client

@task(log_prints=True)
def load_data():
    X_train, y_train = prepare_data("data/partition/train.parquet")
    X_test, y_test = prepare_data("data/partition/test.parquet")
    return X_train, X_test, y_train, y_test

@task(log_prints=True)
def run_all_experiments(X_train, X_test, y_train, y_test):
    # xgb_run_id, xgb_recall = train_xgb(X_train, y_train, X_test, y_test)
    # lr_run_id, lr_recall = train_lr(X_train, y_train, X_test, y_test)
    # rf_run_id, rf_recall = train_rf(X_train, y_train, X_test, y_test)
    # results = [
    #     ("XGBoost", xgb_run_id, xgb_recall),
    #     ("RandomForest", rf_run_id, rf_recall),
    #     ("LogReg", lr_run_id, lr_recall)
    # ]

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

@task(log_prints=True)
def promote_best_model(winner_id, winner_score):
    client = setup_mlflow()
    bucket_root = "s3://predictive-maintenance-artifacts-victor-obi/mlflow/models"
    model_url = f"runs:/{winner_id}/model"
    try:
        prod_models = client.get_latest_versions(name=experiment_name, stages=["Production"])
        current_prod = prod_models[0]
        prod_run = client.get_run(current_prod.run_id)
        prod_recall = prod_run.data.metrics.get('test_recall', 0.0)
    except:
        prod_recall = 0.0

    print(f"DEBUG: Registering model from S3 path: {model_url}")
    mv = mlflow.register_model(model_url, name=experiment_name)

    if winner_score > prod_recall:
        client.transition_model_version_stage(
            name=experiment_name,
            version=mv.version,
            stage='Production',
            archive_existing_versions=True
        )
    else:
        client.transition_model_version_stage(
            name=experiment_name,
            version=mv.version,
            stage='Archived',
            archive_existing_versions=True
        )
            

@flow(name="Predictive Maintenance Training Flow")
def main_flow():
    pull_dvc_data()
    setup_mlflow()
    X_train, X_test, y_train, y_test = load_data()
    winner_id, winner_score = run_all_experiments(X_train, X_test, y_train, y_test)
    promote_best_model(winner_id, winner_score)



if __name__ == "__main__":
    main_flow()


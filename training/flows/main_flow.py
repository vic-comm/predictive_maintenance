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
experiment_name = "Predictive-Maintenance-Final"
bucket = "s3://predictive-maintenance-artifacts-victor-obi/mlflow"

# @task(log_prints=True)
# def pull_dvc_data():
#     # try:
        
#     #     subprocess.run(["dvc", "pull"], check=True)
#     #     print("Data pulled successfully!")

#     # except subprocess.CalledProcessError as e:
#     #     print(f"DVC Pull Failed: {e}")
#     #     raise e

#     result = subprocess.run(["dvc", "pull", "-v"], capture_output=True, text=True) # -v adds verbose logging
    
#     if result.returncode == 0:
#         print("✅ Data pulled successfully!")
#         print(result.stdout)
#     else:
#         print("DVC Pull Failed!")
#         print("------------- STDOUT -------------")
#         print(result.stdout)
#         print("------------- STDERR -------------")
#         print(result.stderr)
#         print("----------------------------------")
#         raise Exception("DVC Pull failed")
    

# import os
# import subprocess
# from prefect import task

# @task(log_prints=True)
# def pull_dvc_data():
#     print("🔍 Checking credentials...")
    
#     # Get the DagsHub (Office) Keys
#     dags_user = os.getenv("DAGSHUB_USER")
#     dags_token = os.getenv("DAGSHUB_TOKEN")
    
#     if not dags_user or not dags_token:
#         raise Exception("Missing DAGSHUB credentials!")

#     # --- THE MAGIC TRICK ---
#     # We create a copy of the environment variables
#     # and REMOVE the Amazon AWS keys from it.
#     # This prevents DVC from seeing the "House Key".
#     clean_env = os.environ.copy()
#     clean_env.pop("AWS_ACCESS_KEY_ID", None)
#     clean_env.pop("AWS_SECRET_ACCESS_KEY", None)
#     # -----------------------

#     print("⚙️ configuring DVC local credentials...")
#     try:
#         # We pass 'env=clean_env' so DVC only sees the DagsHub keys
#         subprocess.run(["dvc", "remote", "modify", "origin", "--local", "access_key_id", dags_user], check=True, env=clean_env)
#         subprocess.run(["dvc", "remote", "modify", "origin", "--local", "secret_access_key", dags_token], check=True, env=clean_env)
#     except Exception as e:
#         print(f"❌ Failed to configure DVC: {e}")
#         raise e

#     print("🔄 Starting DVC Pull...")
#     # Again, we pass 'env=clean_env' so DVC doesn't get confused by Amazon keys
#     result = subprocess.run(["dvc", "pull", "-v"], capture_output=True, text=True, env=clean_env)
    
#     if result.returncode == 0:
#         print("✅ Data pulled successfully!")
#         print(result.stdout)
#     else:
#         print("❌ DVC Pull Failed!")
#         print(result.stderr)
#         raise Exception("DVC Pull failed")    

import subprocess
from prefect import task

@task(log_prints=True)
def pull_dvc_data():
    print("Starting DVC Pull from S3...")
    
    # No need for clean_env or hidden keys! 
    # It will automatically use the AWS keys from Prefect.
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

@task
def load_data():
    X_train, y_train = prepare_data("data/partition/train.parquet")
    X_test, y_test = prepare_data("data/partition/test.parquet")
    return X_train, X_test, y_train, y_test

@task
def run_all_experiments(X_train, X_test, y_train, y_test):
    client = setup_mlflow()
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
    client = setup_mlflow()
    try:
        prod_models = client.get_latest_versions(name=experiment_name, stages=["Production"])
        current_prod = prod_models[0]
        prod_run = client.get_run(current_prod.run_id)
        prod_recall = prod_run.data.metrics['recall']
    except:
        prod_recall = 0.0

    model_url = f"runs:/{winner_id}/model"
    mv = mlflow.register_model(model_url, name=experiment_name)

    if winner_score > prod_recall:
        client.transition_model_version_stage(
            name=experiment_name,
            version=mv.version,
            stage='Production',
            archive_existing_versions=True
        )
    # else:
    #     client.transition_model_version_stage(
    #         name=experiment_name,
    #         version=mv.version,
    #         stage='Archived',
    #         archive_existing_versions=True
    #     )
            

@flow(name="Predictive Maintenance Training Flow")
def main_flow():
    pull_dvc_data()
    setup_mlflow()
    X_train, X_test, y_train, y_test = load_data()
    winner_id, winner_score = run_all_experiments(X_train, X_test, y_train, y_test)
    promote_best_model(winner_id, winner_score)



# if __name__ == "__main__":
#     # main_flow()
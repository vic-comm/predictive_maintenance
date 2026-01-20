import mlflow
import os
import time
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient
load_dotenv()

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
MODEL_URI = "models:/Predictive-Maintenance-Experiment/Production"


mlflow.set_tracking_uri(TRACKING_URI)

def run_debug():
    client = MlflowClient()
    try:
        start = time.time()
        versions = client.get_latest_versions(name="Predictive-Maintenance-Experiment", stages=["Production"])
        if not versions:
            raise RuntimeError("No model in Production stage")
        mv = versions[0]
        duration = time.time() - start
        print(f" source: {mv.source}")
        print(f" SUCCESS ({duration:.2f}s)")        
    except Exception as e:
        print(f"FAILED: Could not talk to DagsHub.")
        print(f"   Error: {e}")
        print("\nPossible Causes:")
        print("   1. DagsHub is down (500 Error).")
        print("   2. MLFLOW_TRACKING_PASSWORD is missing or wrong.")
        return # Stop here
    # try:
    #     start = time.time()

    #     download_uri = client.get_model_version_download_uri(
    #         "Predictive-Maintenance-Experiment",
    #         mv.version
    #     )

    #     duration = time.time() - start

    #     print(f"SUCCESS ({duration:.2f}s)")
    #     print("   DagsHub resolved a signed artifact URI:")
    #     print(f"   👉 {download_uri}")

    # except Exception as e:
    #     print("❌ FAILED: Registry reachable, but artifact resolution failed.")
    #     print(f"   Error: {e}")
    #     print("\nPossible Causes:")
    #     print("   1. Corrupted model version")
    #     print("   2. DagsHub artifact backend issue")
    #     return
    print("\nTEST 2: Downloading Model from S3...")
    try:
        start = time.time()
        model = mlflow.pyfunc.load_model(MODEL_URI)
        duration = time.time() - start
        
        print(f"SUCCESS ({duration:.2f}s)")
        print(f"   Model loaded into memory correctly.")
        
    except Exception as e:
        print(f"FAILED: DagsHub worked, but S3 download failed.")
        print(f"   Error: {e}")
        print("\nPossible Causes:")
        print("   1. Your AWS Credentials are expired.")
        print("   2. The S3 path provided by DagsHub is wrong.")

if __name__ == "__main__":
    run_debug()
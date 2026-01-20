import mlflow
import os
import time
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient

load_dotenv()

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
# We still need this to find the S3 path initially
REGISTRY_URI = "models:/Predictive-Maintenance-Experiment/Production"

mlflow.set_tracking_uri(TRACKING_URI)

def run_debug():
    client = MlflowClient()
    
    print("-" * 50)
    print("📡 STEP 1: Fetching S3 Path from Registry...")
    s3_source = None
    
    try:
        # Get the latest version details
        versions = client.get_latest_versions(name="Predictive-Maintenance-Experiment", stages=["Production"])
        if not versions:
            raise RuntimeError("No model in Production stage")
            
        mv = versions[0]
        s3_source = mv.source
        
        # HACK: The registry might say ".../artifacts/model", but looking at your
        # screenshots, the files (MLmodel, model.pkl) are directly inside ".../artifacts".
        # We try the exact source first, but keep this in mind if it fails.
        
        print(f"✅ FOUND SOURCE: {s3_source}")
        
    except Exception as e:
        print(f"❌ FAILED to look up model version: {e}")
        return

    print("\n💾 STEP 2: Downloading Direct from S3 (Bypassing DagsHub API)...")
    try:
        start = time.time()
        
        # CRITICAL CHANGE: We use the s3_source directly!
        # This uses your local AWS credentials and talks ONLY to AWS.
        print(f"   Attempting to load from: {s3_source}")
        model = mlflow.pyfunc.load_model(s3_source)
        
        duration = time.time() - start
        print(f"✅ SUCCESS ({duration:.2f}s)")
        print("   Model loaded into memory! DagsHub API was bypassed.")
        
    except Exception as e:
        print(f"❌ S3 LOAD FAILED.")
        print(f"   Error: {e}")
        print("\nTroubleshooting S3 Path:")
        # If the specific /model folder fails, try the parent directory
        if "key does not exist" in str(e).lower() or "404" in str(e):
             parent_path = s3_source.rsplit('/', 1)[0]
             print(f"   ⚠️ Trying parent folder: {parent_path} ...")
             try:
                 model = mlflow.pyfunc.load_model(parent_path)
                 print(f"   ✅ SUCCESS! Loaded from parent folder: {parent_path}")
             except Exception as e2:
                 print(f"   ❌ Parent folder failed too: {e2}")

if __name__ == "__main__":
    run_debug()
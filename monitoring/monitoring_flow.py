import pandas as pd
import boto3
from prefect import flow, task
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
import json

# Configuration
NAMESPACE = "PredictiveMaintenance"
METRIC_NAME = "DriftScore"

@task(name="Fetch & Calculate Drift")
def calculate_drift():
    # 1. Load Data (Simulated)
    # In real life: Reference = S3 Training Data, Current = Yesterday's Database dump
    reference = pd.read_parquet("data/partition/train.parquet") 
    current = pd.read_parquet("data/partition/test.parquet")
    
    # SIMULATION: Corrupt the data to trigger the alarm (Comment this out later)
    current['air_temperature_k'] = current['air_temperature_k'] * 1.5 
    
    # 2. Run Evidently
    print("Running drift calculation...")
    drift_report = Report(metrics=[DataDriftPreset()])
    drift_report.run(reference_data=reference, current_data=current)
    
    # 3. Extract the exact score (Number of drifted columns share)
    # Evidently outputs complex JSON, we need one simple number for CloudWatch
    json_output = drift_report.json()
    data = json.loads(json_output)
    
    # Get share of drifted columns (0.0 to 1.0)
    drift_share = data['metrics'][0]['result']['share_of_drifted_columns']
    print(f"Drift Score: {drift_share}")
    
    return drift_share

@task(name="Push to CloudWatch")
def push_metric(value):
    print(f"Pushing metric {value} to CloudWatch...")
    cloudwatch = boto3.client('cloudwatch', region_name='us-east-1')
    
    cloudwatch.put_metric_data(
        Namespace=NAMESPACE,
        MetricData=[
            {
                'MetricName': METRIC_NAME,
                'Value': value,
                'Unit': 'Count',
                'StorageResolution': 60
            },
        ]
    )
    print("✅ Metric pushed successfully.")

@flow(name="Daily Monitoring Flow")
def monitoring_flow():
    drift_score = calculate_drift()
    push_metric(drift_score)

if __name__ == "__main__":
    monitoring_flow()
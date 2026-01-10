import base64
import json
import boto3
import mlflow
import dagshub
import mlflow.sklearn
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv()
LOGGED_MODEL = f"models:/predictive-maintenance-prediction/Production"
OUTPUT_STREAM_NAME = "predictive-maintenance-predictions"
kinesis_client = boto3.client('kinesis')

tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
if tracking_uri:
    mlflow.set_tracking_uri(tracking_uri)
try:
    model = mlflow.sklearn.load_model(LOGGED_MODEL)
    print("Model loaded successfully.")
except Exception as e:
    print(f"CRITICAL ERROR: Could not load model. {str(e)}")
    model = None

def decode_kinesis_data(record):
    try:
        payload = base64.b64decode(record['kinesis']['data'])
        json_str = payload.decode('utf-8')
        return json.loads(json_str)
    except Exception as e:
        return None

def prepare_features(input_data):
    df = pd.DataFrame([input_data])
    required_cols = [
        'Air_temperature_K', 'Process_temperature_K', 'Rotational_speed_rpm', 
        'Torque_Nm', 'Tool_wear_min', 'Type_encoded'
    ]
    for col in required_cols:
        if col not in df.columns:
            df[col] = 0 
    return df[required_cols]

def lambda_handler(event, context):
    if model is None:
        return {'statusCode': 500, 'body': json.dumps("Model not initialized")}

    predictions = []
    
    # Handle Kinesis Batch
    for record in event.get('Records', []):
        try:
            data = decode_kinesis_data(record)
            if not data: continue

            features = prepare_features(data)
            pred_probs = model.predict_proba(features)
            failure_prob = float(pred_probs[0][1])
            
            prediction = 1 if failure_prob > 0.5 else 0
            
            result = {
                'input_id': data.get('UDI', 'Unknown'), 
                'prediction': prediction,
                'probability': round(failure_prob, 4),
                'status': 'Danger' if prediction == 1 else 'Normal'
            }
            
            print(f"PREDICTION: {json.dumps(result)}")
            try:
                kinesis_client.put_record(
                    StreamName=OUTPUT_STREAM_NAME,
                    Data=json.dumps(result) + "\n", 
                    PartitionKey=str(result['input_id'])
                )
            except Exception as e:
                print(f"Failed to write to output stream: {e}")

            predictions.append(result)

        except Exception as e:
            print(f"Error processing record: {str(e)}")

    return {
        'statusCode': 200,
        'body': json.dumps({'processed_count': len(predictions)})
    }
import base64
import json
import boto3
import mlflow
import dagshub
import pandas as pd
import os
from mlflow.tracking import MlflowClient
from dotenv import load_dotenv

load_dotenv()
EXPERIMENT_NAME = "Predictive-Maintenance-Experiment"
STAGE="Production"
MODEL_URI=F"models:/{EXPERIMENT_NAME}/{STAGE}"
# MODEL_URI = "models:/Predictive-Maintenance-Experiment/Production"
OUTPUT_STREAM_NAME = "predictive-maintenance-predictions"
kinesis_client = boto3.client('kinesis')


def decode_kinesis_data(record):
    try:
        payload = base64.b64decode(record['kinesis']['data'])
        json_str = payload.decode('utf-8')
        return json.loads(json_str)
    except Exception as e:
        return None

def predict_proba_robust(model, features):
    impl = model._model_impl
    
    if hasattr(impl, "sklearn_model"):
        return float(impl.sklearn_model.predict_proba(features)[0][1])

    elif hasattr(impl, "xgb_model"):
        xgb_native = impl.xgb_model
        
        if hasattr(xgb_native, "predict_proba"):
            return float(xgb_native.predict_proba(features)[0][1])
            
       
        import xgboost as xgb
        dmatrix = xgb.DMatrix(features)
        return float(xgb_native.predict(dmatrix)[0])

   
    return float(model.predict(features)[0])

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

model = None
def lambda_handler(event, context):

    global model
    
    if model is None:
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
            client = MlflowClient()
        versions = client.get_latest_versions(name=EXPERIMENT_NAME, stages=[STAGE])
        mv = versions[0]
        source = mv.source
        # model = mlflow.pyfunc.load_model(source)
        try:
            model = mlflow.pyfunc.load_model(source)
        except Exception as e:
            print(f"Direct load failed: {e}. Trying parent directory...")
            # Try removing the last part of the path (e.g., remove '/model')
            if source.endswith("/model"):
                parent_source = source.rsplit('/', 1)[0]
                print(f"Retrying with parent source: {parent_source}")
                model = mlflow.pyfunc.load_model(parent_source)
            else:
                raise e

    predictions = []
    
    # Handle Kinesis Batch
    for record in event.get('Records', []):
        try:
            data = decode_kinesis_data(record)
            if not data: continue

            features = prepare_features(data)
            failure_prob = predict_proba_robust(model, features)
            prediction = 1 if failure_prob > 0.5 else 0
            # pred_probs = model.predict(features)
            # failure_prob = float(pred_probs[0][1])
            
            # prediction = 1 if failure_prob > 0.5 else 0
            
            result = {
                'input_id': data.get('UDI', 'Unknown'), 
                'prediction': prediction,
                'probability': round(failure_prob, 4),
                'status': 'Danger' if prediction == 1 else 'Normal',
                'model_version': 'Production'
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


# # Go back to root
# cd ../..

# # Tag the image (use your copied ECR URL)
# docker tag predictive-maintenance-lambda:latest 852100867623.dkr.ecr.us-east-1.amazonaws.com/predictive-maintenance-lambda:latest

# # Push
# docker push 852100867623.dkr.ecr.us-east-1.amazonaws.com/predictive-maintenance-lambda:latest
# ecr_repository_url = "852100867623.dkr.ecr.us-east-1.amazonaws.com/predictive-maintenance-lambda"

# cd streaming/prediction

# # 1. Build (Clean x86 build)
# docker build --platform linux/amd64 --provenance=false -t predictive-maintenance-lambda .

# # 2. Tag
# docker tag predictive-maintenance-lambda:latest 852100867623.dkr.ecr.us-east-1.amazonaws.com/predictive-maintenance-lambda:latest

# # 3. Push
# docker push 852100867623.dkr.ecr.us-east-1.amazonaws.com/predictive-maintenance-lambda:latest

# # 4. Force Update Lambda
# aws lambda update-function-code \
#   --function-name predictive-maintenance-function-tf \
#   --image-uri 852100867623.dkr.ecr.us-east-1.amazonaws.com/predictive-maintenance-lambda:latest

# cd ../infrastructure
# terraform apply -var="deploy_lambda=true"

# aws logs tail /aws/lambda/predictive-maintenance-function-tf --follow --format short

# docker run -it --rm \
#   -e AWS_ACCESS_KEY_ID=$(aws configure get aws_access_key_id) \
#   -e AWS_SECRET_ACCESS_KEY=$(aws configure get aws_secret_access_key) \
#   -e AWS_DEFAULT_REGION=us-east-1 \
#   sensor-producer

# cd streaming/prediction
# docker build --platform linux/amd64 --provenance=false -t predictive-maintenance-lambda .
# docker tag predictive-maintenance-lambda:latest 852100867623.dkr.ecr.us-east-1.amazonaws.com/predictive-maintenance-lambda:latest
# docker push 852100867623.dkr.ecr.us-east-1.amazonaws.com/predictive-maintenance-lambda:latest


# ecr_repository_url = "852100867623.dkr.ecr.us-east-1.amazonaws.com/predictive-maintenance-lambda"
# input_stream_name = "predictive-maintenance-stream"
# output_stream_name = "predictive-maintenance-predictions"
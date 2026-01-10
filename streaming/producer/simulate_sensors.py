import json
import time
import boto3
import pandas as pd
import random
import os
import re

KINESIS_STREAM_NAME = "predictive-maintenance-stream"

DATA_FILE_PATH = "../data/partition/stream.csv" 

def get_kinesis_client():
    return boto3.client('kinesis', region_name='us-east-1')

def prepare_record(row):
    type_mapping = {'L': 0, 'M': 1, 'H': 2}
    type_val = row.get('Type', 'L')
    data = {
        "UDI": int(row.get('UDI', random.randint(10000, 99999))),
        "Air_temperature_K": float(row.get('Air_temperature_K', 0)),
        "Process_temperature_K": float(row.get('Process_temperature_K', 0)),
        "Rotational_speed_rpm": int(row.get('Rotational_speed_rpm', 0)),
        "Torque_Nm": float(row.get('Torque_Nm', 0)),
        "Tool_wear_min": int(row.get('Tool_wear_min', 0)),
        "Type_encoded": type_mapping.get(type_val, 0)
    }
    return data

def main():
    print(f"Loading data from {DATA_FILE_PATH}...")
    
    if not os.path.exists(DATA_FILE_PATH):
        print(f"ERROR: File not found at {DATA_FILE_PATH}")
        print("   Please check the path. Your current working directory is:", os.getcwd())
        return


    try:
        df = pd.read_csv(DATA_FILE_PATH)
        print(f"Loaded {len(df)} rows ready for streaming.")
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return
    
    mapping = {'L': 0, 'M': 1, 'H': 2}
    df['Type_encoded'] = df['Type'].map(mapping).astype(int)
    drop_cols = ['Type', 'UID', 'Product ID', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']
    df = df.drop(drop_cols, axis=1, errors='ignore')
    df.columns = [re.sub(r"[\[\]<]", "", col) for col in df.columns]

    df.columns = [col.replace(" ", "_") for col in df.columns] 
    kinesis = get_kinesis_client()
    
    print(f"Streaming to Kinesis: {KINESIS_STREAM_NAME}...")    
    for index, row in df.iterrows():
        try:
            record = prepare_record(row)
            kinesis.put_record(
                StreamName=KINESIS_STREAM_NAME,
                Data=json.dumps(record),
                PartitionKey=str(record['UDI'])
            )
            
            print(f"-> Sent record #{index} | ID: {record['UDI']}")
            time.sleep(0.5) 
            
        except Exception as e:
            print(f"Error sending record #{index}: {e}")
            

if __name__ == "__main__":
    main()
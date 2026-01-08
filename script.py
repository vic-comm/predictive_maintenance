import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('data/ai4i2020.csv')

train, temp = train_test_split(df, train_size=0.7, random_state=20)
test, stream = train_test_split(temp, test_size=0.5, random_state=20)

train.to_parquet("data/partition/train.parquet")
test.to_parquet("data/partition/test.parquet")
stream.to_csv("data/partition/stream.csv", index=False)


# predictive-maintenance/
# │
# ├── .github/workflows/          # CI/CD Pipelines
# ├── data/                       # Local data (gitignored)
# │   ├── raw/                    # The AI4I CSV file
# │   └── partition/              # Train/Test/Stream splits
# │
# ├── infrastructure/             # The "Cloud" (Terraform)
# │   ├── modules/
# │   ├── main.tf                 # Defines Kinesis, Lambda, S3
# │   └── variables.tf
# │
# ├── streaming/                  # The "Real-Time" Code
# │   ├── producer/               # The Data Simulator
# │   │   ├── simulate_sensors.py # Reads CSV, hits API
# │   │   └── Dockerfile
# │   ├── ingestion/              # The Gateway
# │   │   ├── app.py              # FastAPI app
# │   │   └── Dockerfile
# │   └── prediction/             # The Brain
# │       ├── lambda_function.py  # Inference Logic
# │       ├── model.pkl           # (Loaded during build)
# │       └── Dockerfile          # Crucial for "Fat Lambda"
# │
# ├── training/                   # The "Batch" Code
# │   ├── flows/                  # Prefect Flows
# │   │   ├── train_model.py
# │   │   └── monitor_drift.py
# │   └── notebooks/              # Exploration & Experiments
# │
# ├── Makefile                    # Shortcuts (make setup, make deploy)
# ├── pyproject.toml              # Dependencies (Poetry/Pip)
# └── README.md
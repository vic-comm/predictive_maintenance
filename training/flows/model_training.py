import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, recall_score, f1_score, accuracy_score, roc_auc_score, confusion_matrix, roc_curve, auc
import xgboost as xgb
from sklearn.feature_extraction import DictVectorizer
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope
from sklearn.linear_model import LogisticRegression
import mlflow
import mlflow.sklearn
import numpy as np

def prepare_data(path):
    if path.endswith('.csv'):
        data = pd.read_csv(path)
    elif path.endswith('.parquet'):
        data = pd.read_parquet(path)
    
    mapping = {'L': 0, 'M': 1, 'H': 2}
    data['Type_encoded'] = data['Type'].map(mapping).astype(int)
    drop_cols = ['Type', 'UDI', 'Product ID', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']
    data = data.drop(drop_cols, axis=1, errors='ignore')
    
    data.columns = [re.sub(r"[\[\]<]", "", col) for col in data.columns]
    data.columns = [col.replace(" ", "_") for col in data.columns] 

    target = 'Machine_failure'
    
    if target in data.columns:
        y = data[target]
        if isinstance(y, pd.DataFrame):
            y = y.iloc[:, 0]
        
        y = y.to_numpy().ravel().astype(int)
        
        X = data.drop([target], axis=1)
        return X, y
    else:
        return data, None
    

def train_lr(X_train, y_train, X_test, y_test):
    y_test = np.array(y_test).ravel()
    def objective_lr(params):
        mlflow.sklearn.autolog(disable=True)     
        with mlflow.start_run(nested=True):
            mlflow.log_params(params)
            
            lr = LogisticRegression(
                C=params['C'],
                solver=params['solver'],
                class_weight='balanced', 
                max_iter=1000,
                random_state=42
            )
            
            lr.fit(X_train, y_train)
            
            y_pred = lr.predict(X_test)
            y_prob = lr.predict_proba(X_test)[:, 1]
            
            test_recall = recall_score(y_test, y_pred)
            test_roc_auc = roc_auc_score(y_test, y_prob)
            test_accuracy = accuracy_score(y_test, y_pred)
            test_f1 = f1_score(y_test, y_pred)
            mlflow.log_metric('test_recall', test_recall)
            mlflow.log_metric('test_roc_auc', test_roc_auc)
            mlflow.log_metric('test_f1_score', test_f1)
            mlflow.log_metric('test_accuracy', test_accuracy)

            return {'loss': -test_roc_auc, 'status': STATUS_OK}
    
    search_space_lr = {
        'C': hp.loguniform('C', -4, 2), 
        'solver': hp.choice('solver', ['liblinear', 'lbfgs'])
    }
    
    best_result_lr = fmin(
        fn=objective_lr,
        space=search_space_lr,
        algo=tpe.suggest,
        max_evals=10,
        trials=Trials()
    )
    
    final_params_lr = best_result_lr.copy()
    solver_list = ['liblinear', 'lbfgs']
    final_params_lr['solver'] = solver_list[final_params_lr['solver']]
    
    with mlflow.start_run() as run:
        mlflow.log_params(final_params_lr)
        
        lr_final = LogisticRegression(**final_params_lr, class_weight='balanced', max_iter=1000)
        lr_final.fit(X_train, y_train)
        y_prob = lr_final.predict_proba(X_test)[:, 1] 
        y_pred = (y_prob > 0.5).astype(int)
            
        test_recall = recall_score(y_test, y_pred)
        print(f"DEBUG: Attempting to save LR model for run {run.info.run_id}...")
        mlflow.sklearn.log_model(sk_model=lr_final, artifact_path="model")
        model_uri = mlflow.get_artifact_uri("model")
        print(f"DEBUG: Model successfully saved to: {model_uri}")
        print("Logistic Regression Champion Saved.")
        # ARTIFACT 1: Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Normal', 'Fail'], 
                    yticklabels=['Normal', 'Fail'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        
        # Save locally, then upload
        plt.savefig("confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png")
        plt.close()

        # ARTIFACT 2: ROC Curve 
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc_val = auc(fpr, tpr)
        
        plt.figure(figsize=(6, 4))
        plt.plot(fpr, tpr, label=f'AUC = {roc_auc_val:.2f}')
        plt.plot([0, 1], [0, 1], 'k--') # Random guess line
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc='lower right')
        
        plt.savefig("roc_curve.png")
        mlflow.log_artifact("roc_curve.png")
        plt.close()

        # ARTIFACT 3: Feature Importance 
        importances = np.abs(lr_final.coef_[0])
        feature_names = X_train.columns
        indices = np.argsort(importances)[::-1]
         
        plt.figure(figsize=(10, 6))
        sns.barplot(x=importances[indices], y=feature_names[indices])
        plt.title('Feature Importance (Coefficients)')
        plt.savefig("feature_importance.png")
        mlflow.log_artifact("feature_importance.png")
        plt.close()
    return run.info.run_id, test_recall
    

def train_xgb(X_train, y_train, X_test, y_test):
    y_test = np.array(y_test).ravel()
    def objective_xgb(params):
        params['objective'] = 'binary:logistic'
        params['scale_pos_weight'] = 27.5 
        
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)
        
        mlflow.xgboost.autolog(disable=True)
        
        with mlflow.start_run(nested=True):
            mlflow.log_params(params)
            
            booster = xgb.train(
                params=params,
                dtrain=dtrain,
                num_boost_round=100,
                evals=[(dtest, 'validation')],
                early_stopping_rounds=10,
                verbose_eval=False
            )
            
            y_prob = booster.predict(dtest)      
            y_pred = (y_prob > 0.5).astype(int)  
            
            test_recall = recall_score(y_test, y_pred)
            test_f1 = f1_score(y_test, y_pred)
            test_roc_auc = roc_auc_score(y_test, y_prob)
            test_accuracy = accuracy_score(y_test, y_pred)

            mlflow.log_metric('test_recall', test_recall)
            mlflow.log_metric('test_f1_score', test_f1) 
            mlflow.log_metric('test_roc_auc', test_roc_auc)
            mlflow.log_metric('test_accuracy', test_accuracy)

            return {'loss': -test_roc_auc, 'status': STATUS_OK}
    
    search_space_xgb = {
        'max_depth': scope.int(hp.quniform('max_depth', 4, 20, 1)),
        'learning_rate': hp.loguniform('learning_rate', -3, 0),
        'reg_alpha': hp.loguniform('reg_alpha', -5, -1),
        'reg_lambda': hp.loguniform('reg_lambda', -6, -1),
        'min_child_weight': hp.loguniform('min_child_weight', -1, 3),
        'seed': 42
    }
    
    best_result_xgb = fmin(
        fn=objective_xgb,
        space=search_space_xgb,
        algo=tpe.suggest,
        max_evals=10,
        trials=Trials()
    )
    
    final_params = best_result_xgb.copy()
    final_params['max_depth'] = int(final_params['max_depth'])
    final_params['objective'] = 'binary:logistic'
    final_params['scale_pos_weight'] = 27.5
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    
    with mlflow.start_run() as run:
        mlflow.log_params(final_params)
        
        xg_model = xgb.train(
            params=final_params, 
            dtrain=dtrain, 
            evals=[(dtest, 'validation')],
            num_boost_round=100,
            early_stopping_rounds=10
        )
        y_prob = xg_model.predict(dtest)      
        y_pred = (y_prob > 0.5).astype(int)  
            
        test_recall = recall_score(y_test, y_pred)
        
        mlflow.xgboost.log_model(
            xgb_model=xg_model, 
            artifact_path="model", 
        )
        model_uri = mlflow.get_artifact_uri("model")
        print(f"DEBUG: Model successfully saved to: {model_uri}")
        print(f"XGBoost Champion Saved. Run ID: {run.info.run_id}")
        # ARTIFACT 1: Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Normal', 'Fail'], 
                    yticklabels=['Normal', 'Fail'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        
        plt.savefig("confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png")
        plt.close()

        # ARTIFACT 2: ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc_val = auc(fpr, tpr)
        
        plt.figure(figsize=(6, 4))
        plt.plot(fpr, tpr, label=f'AUC = {roc_auc_val:.2f}')
        plt.plot([0, 1], [0, 1], 'k--') # Random guess line
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc='lower right')
        
        plt.savefig("roc_curve.png")
        mlflow.log_artifact("roc_curve.png")
        plt.close()

        # ARTIFACT 3: Feature Importance 
        importance_dict = xg_model.get_score(importance_type='gain')
        
        if importance_dict:
            # Convert to lists for plotting
            features = list(importance_dict.keys())
            scores = list(importance_dict.values())
            
            # Sort by score (Descending)
            indices = np.argsort(scores)[::-1]
            sorted_features = [features[i] for i in indices]
            sorted_scores = [scores[i] for i in indices]
            
            plt.figure(figsize=(10, 6))
            sns.barplot(x=sorted_scores, y=sorted_features)
            plt.title('XGBoost Feature Importance (Gain)')
            plt.xlabel('Importance Score')
            
            plt.savefig("feature_importance.png")
            mlflow.log_artifact("feature_importance.png")
            plt.close()
    return run.info.run_id, test_recall
    
def train_rf(X_train, y_train, X_test, y_test):
    y_test = np.array(y_test).ravel()
    def objective_rf(params):
        mlflow.sklearn.autolog(disable=True)
        
        with mlflow.start_run(nested=True):
            mlflow.log_params(params)
            
            rf = RandomForestClassifier(
                n_estimators=int(params['n_estimators']),
                max_depth=int(params['max_depth']),
                min_samples_split=int(params['min_samples_split']),
                min_samples_leaf=int(params['min_samples_leaf']),
                criterion=params['criterion'],
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )
            
            rf.fit(X_train, y_train)
            
            y_pred = rf.predict(X_test)
            y_prob = rf.predict_proba(X_test)[:, 1] 

            test_recall = recall_score(y_test, y_pred)
            test_f1 = f1_score(y_test, y_pred)
            test_roc_auc = roc_auc_score(y_test, y_prob)
            test_accuracy = accuracy_score(y_test, y_pred)
            
            mlflow.log_metric('test_recall', test_recall)
            mlflow.log_metric('test_f1_score', test_f1)
            mlflow.log_metric('test_roc_auc', test_roc_auc)     
            mlflow.log_metric('test_accuracy', test_accuracy)       

            return {'loss': -test_roc_auc, 'status': STATUS_OK}

    search_space_rf = {
        'n_estimators': scope.int(hp.quniform('n_estimators', 50, 500, 10)),
        'max_depth': scope.int(hp.quniform('max_depth', 5, 50, 1)),
        'min_samples_split': scope.int(hp.quniform('min_samples_split', 2, 20, 1)),
        'min_samples_leaf': scope.int(hp.quniform('min_samples_leaf', 1, 10, 1)),
        'criterion': hp.choice('criterion', ['gini', 'entropy'])
    }

    best_result = fmin(
        fn=objective_rf,
        space=search_space_rf,
        algo=tpe.suggest,
        max_evals=10,
        trials=Trials()
    )

    criteria_list = ['gini', 'entropy']
    final_params = {
        'n_estimators': int(best_result['n_estimators']),
        'max_depth': int(best_result['max_depth']),
        'min_samples_split': int(best_result['min_samples_split']),
        'min_samples_leaf': int(best_result['min_samples_leaf']),
        'criterion': criteria_list[best_result['criterion']]
    }

    print(f"Training Final RF with params: {final_params}")

    with mlflow.start_run() as run:
        mlflow.log_params(final_params)
        
        rf_final = RandomForestClassifier(**final_params, class_weight='balanced', random_state=42)
        rf_final.fit(X_train, y_train)
        y_prob = rf_final.predict_proba(X_test)[:, 1]
            
        y_pred = rf_final.predict(X_test) 
        test_recall = recall_score(y_test, y_pred)
        test_roc_auc = roc_auc_score(y_test, y_prob)
        test_accuracy = accuracy_score(y_test, y_pred)
        
        mlflow.log_metric("test_recall", test_recall)
        mlflow.log_metric('test_roc_auc', test_roc_auc)
        mlflow.log_metric('test_accuracy', test_accuracy)
        mlflow.sklearn.log_model(
            sk_model=rf_final,
            artifact_path="model",
        )
        model_uri = mlflow.get_artifact_uri("model")
        print(f"DEBUG: Model successfully saved to: {model_uri}")
        
        print(f"Random Forest Saved. Run ID: {run.info.run_id}")
        # ARTIFACT 1: 
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Normal', 'Fail'], 
                    yticklabels=['Normal', 'Fail'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        
        # Save locally, then upload
        plt.savefig("confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png")
        plt.close()

        # ARTIFACT 2: ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc_val = auc(fpr, tpr)
        
        plt.figure(figsize=(6, 4))
        plt.plot(fpr, tpr, label=f'AUC = {roc_auc_val:.2f}')
        plt.plot([0, 1], [0, 1], 'k--') # Random guess line
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc='lower right')
        
        plt.savefig("roc_curve.png")
        mlflow.log_artifact("roc_curve.png")
        plt.close()

        # ARTIFACT 3: Feature Importance 
        
        if hasattr(rf_final, 'feature_importances_'): 
            # Get importance
            importances = rf_final.feature_importances_
            # Get feature names (X_train must be a DataFrame)
            feature_names = X_train.columns
            
            # Sort them
            indices = np.argsort(importances)[::-1]
            
            plt.figure(figsize=(10, 6))
            sns.barplot(x=importances[indices], y=feature_names[indices])
            plt.title('Feature Importance')
            plt.xlabel('Importance Score')
            
            plt.savefig("feature_importance.png")
            mlflow.log_artifact("feature_importance.png")
            plt.close()
    return run.info.run_id, test_recall
    


# Below is the **complete, non-negotiable feature checklist** for your **predictive maintenance / predictive analytics system** to be considered **finished** and **production-grade for an ML engineer with strong data skills**.

# No theory. These are **explicit system features** you must have.

# ---

# # Definition of “Complete”

# A **complete system** must:

# * Ingest data continuously or on schedule
# * Transform it into **versioned, validated features**
# * Train, select, deploy, and serve models
# * Expose predictions to users
# * Monitor itself
# * Retrain automatically when it degrades

# Anything less is a **training platform**, not a production ML system.

# ---

# # REQUIRED FEATURES (FINAL CHECKLIST)

# ## 1. Data ingestion & ETL (mandatory)

# ### Required features

# * Raw data ingestion (batch or streaming)
# * Schema validation (fail fast on bad data)
# * Data freshness checks
# * Separation of:

#   * raw data
#   * cleaned data
#   * feature-ready data

# ### Evidence in code

# * `ingest_raw_data` task
# * `validate_schema` task
# * `clean_transform_data` task
# * Artifacts written to S3 with clear prefixes:

#   * `/raw/`
#   * `/processed/`
#   * `/features/`

# ### Tools

# * Prefect
# * Python
# * S3
# * Optional: Great Expectations

# ---

# ## 2. Feature pipeline (non-negotiable)

# ### Required features

# * Explicit feature generation step
# * Versioned feature datasets
# * Time-aware feature logic (lags, rolling stats)
# * Feature reuse between training and inference

# ### Evidence in code

# * `build_features.py`
# * Feature metadata logged to MLflow
# * Feature version tied to model version

# ### This is what separates ML engineers from data scientists.

# ---

# ## 3. Model training & selection (already strong)

# ### Required features

# * Multiple candidate models
# * Hyperparameter tuning
# * Metric-driven selection
# * Deterministic reproducibility

# ### Evidence

# * MLflow experiments
# * Promotion logic
# * Reproducible runs via DVC

# You already satisfy this.

# ---

# ## 4. Batch inference (required)

# ### Required features

# * Scheduled inference flow
# * Pull latest production model
# * Generate predictions on new data
# * Persist predictions with timestamps

# ### Evidence in code

# * `batch_inference_flow`
# * Output stored in:

#   * S3
#   * or database table (`predictions`)

# Without this, monitoring is impossible.

# ---

# ## 5. (Optional but strong) Real-time inference

# ### Required features

# * REST API endpoint
# * JSON input schema
# * Low-latency prediction
# * Versioned model loading

# ### Evidence

# * FastAPI service
# * `/predict` endpoint
# * Dockerized deployment


# ---

# ## 6. Monitoring (must be multi-dimensional)

# ### Required features

# You must monitor **all three**:

# 1. **Data drift**

#    * Feature distribution changes
# 2. **Prediction drift**

#    * Output distribution shifts
# 3. **Performance decay**

#    * Metric degradation over time

# ### Evidence

# * Stored predictions + actuals
# * Evidently reports
# * Scheduled monitoring job

# Monitoring only data drift is **not enough**.

# ---

# ## 7. Retraining trigger & loop (mandatory)

# ### Required features

# * Automatic retraining trigger based on:

#   * drift threshold OR
#   * performance drop OR
#   * scheduled cadence
# * Safe promotion logic
# * Rollback capability

# ### Evidence

# * `if drift_detected: retrain_flow()`
# * MLflow stage transitions
# * Historical model retention

# ---

# ## 8. Model serving governance

# ### Required features

# * Production vs staging models
# * Reproducible promotion
# * Audit trail of decisions

# ### Evidence

# * MLflow registry usage
# * Promotion logs
# * Tagged experiments

# You mostly have this — tighten it.

# ---

# ## 9. Business-facing output (mandatory)

# ### Required features

# * Dashboard showing:

#   * Risk scores
#   * Failure probability trends
#   * Model performance over time
# * Filter by machine, time, severity

# ### Evidence

# * Streamlit / Superset app
# * Connected to prediction store

# If no one can see the output, the system is incomplete.

# ---

# ## 10. Observability & reliability

# ### Required features

# * Structured logging
# * Error handling
# * Retry logic
# * Alerts on failure

# ### Evidence

# * Prefect logs
# * Fail-fast validation
# * Alert hooks (even mocked)

# ---

# # What you can explicitly exclude (to save time)

# You do **not** need:

# * Deep learning
# * Streaming inference
# * Kubernetes
# * Custom feature store implementation
# * Advanced cloud networking

# Those are optional upgrades, not requirements.

# ---

# # Final “Is it complete?” test

# Your system is complete **only if** you can answer “yes” to all:

# * Can I rerun any model from scratch months later?
# * Can I explain exactly which features fed a prediction?
# * Can users access predictions without touching code?
# * Will the system retrain itself when it degrades?
# * Can I safely roll back a bad model?

# If any answer is “no”, it’s not done.

# ---

# ## Bottom line

# You are **70–80% there**.

# Finish:

# 1. Feature pipeline
# 2. Batch inference
# 3. Monitoring beyond data drift
# 4. Dashboard
# 5. Automated retraining

# Do that, and this becomes a **credible ML engineer portfolio project**, not a learning exercise.

# If you want, I can:

# * Convert this into a **GitHub issue checklist**
# * Design the **exact Prefect flows**
# * Help you write a **README that sells this properly**

# Say which.


# grafane queries:
# traffick light: filter @message like /PREDICTION/
# | fields message.status
# | sort @timestamp desc
# | limit 1
# # Probability Trend (The "Heartbeat")
# filter @message like /PREDICTION/
# | stats avg(message.probability) as Prob by bin(1m)
# The "Danger" Counter
# filter @message like /PREDICTION/ and message.prediction = 1
# | stats count(*) as FailureCount by bin(1h)

# Here are the exact queries you need to copy-paste into Grafana.

# **Important Pre-requisite:**
# Since your Lambda prints logs in the format `PREDICTION: {...json...}`, we must use the `parse` command to extract the data fields cleanly.

# In Grafana, make sure your Data Source is **CloudWatch**, the Region is `us-east-1`, and the Log Group is set to your Lambda function (e.g., `/aws/lambda/predictive-maintenance-function-tf`).

# ---

# ### Layer 1: The "Operator View" (Current Status)

# #### 1. The "Traffic Light" (Stat Panel)

# * **Goal:** Show "Normal" or "Danger" in a big colored box.
# * **Visualization:** Stat
# * **Settings:**
# * *Color Mode:* Value
# * *Value Mappings:* Normal  Green, Danger  Red


# * **Query:**
# ```sql
# filter @message like /PREDICTION:/
# | parse @message "PREDICTION: *" as payload
# | fields payload.status
# | sort @timestamp desc
# | limit 1

# ```



# #### 2. Live Prediction Probability (Gauge Panel)

# * **Goal:** A speedometer showing risk level.
# * **Visualization:** Gauge
# * **Settings:** Min: 0, Max: 1. Thresholds: Green (0), Yellow (0.5), Red (0.75).
# * **Query:**
# ```sql
# filter @message like /PREDICTION:/
# | parse @message "PREDICTION: *" as payload
# | fields payload.probability
# | sort @timestamp desc
# | limit 1

# ```



# #### 3. Recent Alerts (Table Panel)

# * **Goal:** List only the dangerous machines found in the last 24h.
# * **Visualization:** Table
# * **Query:**
# ```sql
# filter @message like /PREDICTION:/
# | parse @message "PREDICTION: *" as payload
# | filter payload.prediction = 1
# | fields @timestamp, payload.input_id as MachineID, payload.probability as RiskScore
# | sort @timestamp desc

# ```



# ---

# ### Layer 2: The "Engineer View" (Diagnostics)

# #### 4. Sensor Telemetry vs. Thresholds (Time Series)

# * **Missing Data Alert:** currently, your Lambda `result` dictionary **only** saves the prediction, not the input features (Torque, Temperature, etc.).
# * **Fix:** In your `lambda_function.py`, update the result dict to include input data:
# ```python
# # Update this in your python code first!
# result = {
#     ...,
#     'Torque': data.get('Torque_Nm'),
#     'Temp': data.get('Air_temperature_K')
# }

# ```


# * **Query (After Fix):**
# ```sql
# filter @message like /PREDICTION:/
# | parse @message "PREDICTION: *" as payload
# | stats avg(payload.Torque) as Torque, avg(payload.Temp) as Temperature by bin(1m)

# ```



# #### 5. Drift Detection (Time Series)

# * **Goal:** See if the average risk is creeping up over time.
# * **Visualization:** Time Series
# * **Query:**
# ```sql
# filter @message like /PREDICTION:/
# | parse @message "PREDICTION: *" as payload
# | stats avg(payload.probability) as AvgRiskScore by bin(5m)

# ```


# *Insight:* If this line trends upward while your machine count stays the same, your model (or machine health) is drifting.

# ---

# ### Layer 3: The "DevOps View" (System Health)

# For these, you should switch the Grafana Query "Source" from **CloudWatch Logs** to **CloudWatch Metrics** for best accuracy, but here are the Logs versions if you want to keep it simple.

# #### 6. Throughput (Predictions per Minute)

# * **Visualization:** Time Series (Bar chart style)
# * **Query:**
# ```sql
# filter @message like /PREDICTION:/
# | stats count(*) as PredictionsPerMin by bin(1m)

# ```



# #### 7. Error Rate

# * **Visualization:** Time Series (Red bars)
# * **Query:**
# ```sql
# filter @message like /ERROR/ or @message like /Exception/ or @message like /Task timed out/
# | stats count(*) as Errors by bin(5m)

# ```



# #### 8. System Lag (Iterator Age)

# * **Note:** You **cannot** query this from Logs easily. You must use the **Metrics** API.
# * **Grafana Setup:**
# 1. Change **Query Mode** to `Metrics`.
# 2. **Namespace:** `AWS/Lambda`
# 3. **Metric Name:** `IteratorAge`
# 4. **Statistic:** `Maximum`
# 5. **FunctionName:** `predictive-maintenance-function-tf`


# * *Insight:* If this number spikes (e.g., > 10000ms), your Lambda is too slow for the incoming Kinesis data stream.
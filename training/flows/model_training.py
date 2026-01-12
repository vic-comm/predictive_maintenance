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
    drop_cols = ['Type', 'UID', 'Product ID', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']
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
    

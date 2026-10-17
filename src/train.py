import os
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def train():
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "churn_data.csv")
    
    if not os.path.exists(data_path):
        from data.generate_data import generate_churn_dataset
        generate_churn_dataset(output_path=data_path)
        
    df = pd.read_csv(data_path)
    X = df.drop(columns=["churn"])
    y = df["churn"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # MLflow tracking
    mlflow.set_experiment("Customer_Churn_Prediction")
    
    n_estimators = 100
    max_depth = 6
    random_state = 42
    
    with mlflow.start_run(run_name="RandomForest_Baseline"):
        # Log Hyperparameters
        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("train_samples", len(X_train))
        mlflow.log_param("test_samples", len(X_test))
        
        # Train model
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state
        )
        clf.fit(X_train, y_train)
        
        # Evaluate
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_prob)
        
        # Log Metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", auc)
        
        # Log Model artifact in MLflow (using cloudpickle for reliable serialization)
        mlflow.sklearn.log_model(
            sk_model=clf,
            artifact_path="customer_churn_model",
            serialization_format="cloudpickle"
        )
        
        # Save local model file for FastAPI inference
        model_dir = os.path.join(os.path.dirname(__file__), "..", "models")
        os.makedirs(model_dir, exist_ok=True)
        local_model_path = os.path.join(model_dir, "churn_model.pkl")
        joblib.dump(clf, local_model_path)
        
        print("Training Completed Successfully!")
        print(f"Metrics -> Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f} | ROC-AUC: {auc:.4f}")
        print(f"Model saved to: {local_model_path}")

if __name__ == "__main__":
    train()

import mlflow
import mlflow.sklearn
import joblib
import os
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.config import EXPERIMENT_NAME, MODELS_DIR
from src.logger import get_logger
from src.data_preprocessing import load_data, preprocess_data

logger = get_logger("train_pipeline")

def train_and_evaluate(X_train, X_test, y_train, y_test):
    """
    Trains models with tracking via MLflow. Finds the best model based on F1-score.
    """
    mlflow.set_experiment(EXPERIMENT_NAME)
    mlflow.sklearn.autolog()
    
    models = {
        "Logistic_Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random_Forest": RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    best_f1 = 0
    best_model = None
    best_model_name = ""
    
    for model_name, model in models.items():
        logger.info(f"Training {model_name}...")
        
        with mlflow.start_run(run_name=model_name):
            # Set requested tags
            mlflow.set_tags({"project": "college-mlops", "stage": "training"})
            
            # Train the model
            model.fit(X_train, y_train)
            
            # Predictions
            predictions = model.predict(X_test)
            
            # Manually calculate extra metrics although autolog does some of these automatically
            acc = accuracy_score(y_test, predictions)
            prec = precision_score(y_test, predictions)
            rec = recall_score(y_test, predictions)
            f1 = f1_score(y_test, predictions)
            
            logger.info(f"{model_name} metrics - Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
            
            # MLflow autolog logs many standard metrics natively, but we optionally log them here manually if missing
            mlflow.log_metric("custom_accuracy", acc)
            mlflow.log_metric("custom_precision", prec)
            mlflow.log_metric("custom_recall", rec)
            mlflow.log_metric("custom_f1_score", f1)
            
            if f1 > best_f1:
                best_f1 = f1
                best_model = model
                best_model_name = model_name

    return best_model, best_model_name, best_f1

if __name__ == "__main__":
    logger.info("Initiating Training Pipeline")
    
    # Preprocess
    df = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(df)
    
    # Train
    best_model, best_name, best_f1 = train_and_evaluate(X_train, X_test, y_train, y_test)
    logger.info(f"Winner: {best_name} with F1: {best_f1:.4f}")
    
    # Save best model
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, "best_model.pkl")
    joblib.dump(best_model, model_path)
    logger.info(f"Saved best model to {model_path}")

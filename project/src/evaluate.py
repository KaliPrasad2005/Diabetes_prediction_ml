import joblib
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import mlflow

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.config import EXPERIMENT_NAME, MODELS_DIR, ARTIFACTS_DIR
from src.logger import get_logger
from src.data_preprocessing import load_data, preprocess_data

logger = get_logger("evaluate_pipeline")

def evaluate_model():
    """
    Evaluates the best saved model, generates a confusion matrix,
    and logs the image artifact to MLflow.
    """
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    # Load model and data
    model_path = os.path.join(MODELS_DIR, "best_model.pkl")
    if not os.path.exists(model_path):
        logger.error(f"Model not found at {model_path}. Run train.py first.")
        return

    logger.info("Loading dataset and preprocessing for test set...")
    df = load_data()
    _, X_test, _, y_test = preprocess_data(df)
    
    logger.info(f"Loading best model from {model_path}...")
    model = joblib.load(model_path)
    
    with mlflow.start_run(run_name="Evaluation_Reporting") as run:
        mlflow.set_tags({"project": "college-mlops", "stage": "evaluation"})
        
        predictions = model.predict(X_test)
        cm = confusion_matrix(y_test, predictions)
        
        # Plot
        plt.figure(figsize=(8,6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Not Diabetic', 'Diabetic'], 
                    yticklabels=['Not Diabetic', 'Diabetic'])
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.title('Confusion Matrix')
        
        # Save image
        os.makedirs(ARTIFACTS_DIR, exist_ok=True)
        cm_path = os.path.join(ARTIFACTS_DIR, "confusion_matrix.png")
        plt.savefig(cm_path)
        plt.close()
        
        logger.info(f"Confusion matrix image saved to {cm_path}")
        
        # Log artifact to MLflow
        mlflow.log_artifact(cm_path, "evaluation_plots")
        logger.info("Confusion matrix explicitly logged to MLflow successfully.")

if __name__ == "__main__":
    evaluate_model()

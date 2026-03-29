import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os
import sys

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.config import DATA_PATH, TARGET_COLUMN, MODELS_DIR
from src.logger import get_logger

logger = get_logger("data_preprocessing")

def load_data(filepath=DATA_PATH):
    """Loads the dataset from the specified path."""
    logger.info(f"Loading data from {filepath}")
    try:
        df = pd.read_csv(filepath)
        logger.info(f"Dataset loaded with shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise e

def preprocess_data(df):
    """
    Preprocesses the data by splitting features and target, 
    applying StandardScaler, and saving the scaler.
    """
    logger.info("Starting preprocessing")
    
    # Define features and target
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    
    # Handle zeros which actually mean missing values in this specific dataset
    cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for col in cols_with_zeros:
        # Replace 0 with median of the column
        median_val = X[X[col] != 0][col].median()
        X[col] = X[col].replace(0, median_val)
        
    logger.info("Handled zero values in logical features.")

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    logger.info("Train test split completed.")
    
    # Apply standard scaling
    logger.info("Applying StandardScaler to features.")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save the scaler for inference
    os.makedirs(MODELS_DIR, exist_ok=True)
    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
    joblib.dump(scaler, scaler_path)
    logger.info(f"Scaler saved to {scaler_path}")
    
    # Convert scaled arrays back to DataFrames 
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)
    
    return X_train_scaled, X_test_scaled, y_train, y_test

if __name__ == "__main__":
    df = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(df)
    logger.info("Data preprocessing completed successfully.")

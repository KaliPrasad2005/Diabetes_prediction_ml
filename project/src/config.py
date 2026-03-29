import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data paths
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "kaggle_diabetes.csv")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "artifacts")

# MLflow config
EXPERIMENT_NAME = "Diabetes Prediction MLOps"

# Target column for classification
TARGET_COLUMN = "Outcome"

# Ensure directories exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

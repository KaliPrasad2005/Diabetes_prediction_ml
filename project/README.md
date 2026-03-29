# Diabetes Prediction End-to-End System

## 1. Project Overview
A production-ready machine learning system that predicts the likelihood of diabetes based on patient clinical metrics. This project implements a full MLOps pipeline using MLflow, an interactive Streamlit frontend, and a highly portable Docker container setup.

## 2. Architecture
The system consists of 5 main layers:
- **Data Layer**: Robust data preprocessing, zero-handling, and feature scaling (`src/data_preprocessing.py`).
- **Training Layer**: Parallel model training (Logistic Regression & Random Forest) tracked with MLflow autologging, selecting the best model by F1-Score (`src/train.py`).
- **Evaluation Layer**: Systematic model evaluation, visualization through confusion matrices, and logging artifacts to MLflow (`src/evaluate.py`).
- **Serving Layer**: An interactive frontend built for medical intake assessment capable of presenting instant predictions (`app/app.py`).
- **Deployment Layer**: Containerized via Docker to standardize and abstract the runtime execution (`Dockerfile`).

## 3. Tech Stack
- **Language**: Python 3.9
- **Libraries**: scikit-learn, pandas, numpy, matplotlib, seaborn
- **MLOps**: MLflow
- **Frontend**: Streamlit
- **DevOps**: Docker

## 4. MLflow Usage (Training)
To run the training pipeline and let MLflow track your experiments, navigate to the `project` directory and execute:
```bash
# (Optional) Set up a virtual environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run the training pipeline
python src/train.py

# Run model evaluation
python src/evaluate.py
```

## 5. Streamlit Usage
To run the Streamlit application natively in your environment:
```bash
streamlit run app/app.py
```

## 6. Docker Instructions
To build and run the entire application container:

**Build Image:**
```bash
docker build -t diabetes-app .
```

**Run Container:**
```bash
docker run -p 8501:8501 diabetes-app
```
Then navigate to http://localhost:8501 in your browser.

## 7. MLflow UI
To explore the tracked MLflow runs, logged tags, and artifacts:
```bash
mlflow ui
```
Then open http://localhost:5000 in your browser.

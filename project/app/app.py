import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys

# Ensure project root is essentially recognized
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

st.set_page_config(
    page_title="Diabetes Prediction App",
    page_icon="🩺",
    layout="centered"
)

# Custom styling for premium look
st.markdown("""
<style>
    .stButton>button {
        color: white;
        background-color: #ff4b4b;
        border-radius: 8px;
        width: 100%;
        height: 50px;
        font-weight: bold;
        font-size: 18px;
        transition: 0.3s all;
    }
    .stButton>button:hover {
        background-color: #ff2b2b;
        box-shadow: 0px 4px 10px rgba(255, 75, 75, 0.4);
    }
    .stAlert {
        border-radius: 8px;
    }
    h1, h2, h3 {
        color: #1f2937;
    }
    [data-testid="stAppViewContainer"] {
        background-color: #f9fafb;
    }
    [data-testid="stSidebar"] {
        background-color: #ffffff;
    }
</style>
""", unsafe_allow_html=True)

# Application Header
st.title("🩺 Diabetes Prediction App")
st.markdown("""
Welcome to the **MLOps Diabetes Prediction System**. This tool leverages machine learning models tracked over MLflow to assess clinical metrics and predict the likelihood of diabetes.

Please enter the patient records below:
""")

st.divider()

# Model and Scaler Loader
@st.cache_resource
def load_ml_assets():
    # Defining paths manually relative to current dir, running under project/ folder
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, "models", "best_model.pkl")
    scaler_path = os.path.join(base_dir, "models", "scaler.pkl")
    
    try:
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            st.warning("Models not found! Waiting for the MLOps pipeline to be trained.")
            return None, None
            
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

model, scaler = load_ml_assets()

if model and scaler:
    with st.form("prediction_form"):
        st.subheader("Patient Clinical Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1, step=1, 
                                          help="Number of times pregnant")
            glucose = st.number_input("Glucose", min_value=0.0, max_value=300.0, value=100.0, step=1.0, 
                                      help="Plasma glucose concentration (2 hours in oral glucose tolerance test)")
            blood_pressure = st.number_input("Blood Pressure", min_value=0.0, max_value=200.0, value=72.0, step=1.0, 
                                             help="Diastolic blood pressure (mm Hg)")
            skin_thickness = st.number_input("Skin Thickness", min_value=0.0, max_value=100.0, value=35.0, step=1.0, 
                                             help="Triceps skin fold thickness (mm)")
        with col2:
            insulin = st.number_input("Insulin", min_value=0.0, max_value=900.0, value=80.0, step=1.0, 
                                      help="2-Hour serum insulin (mu U/ml)")
            bmi = st.number_input("BMI", min_value=0.0, max_value=100.0, value=32.0, step=0.1, 
                                  help="Body mass index (weight in kg/(height in m)^2)")
            dpf = st.number_input("Diabetes Pedigree Function", min_value=0.000, max_value=3.000, value=0.3725, step=0.001, 
                                  help="Diabetes pedigree function scoring genetic risk")
            age = st.number_input("Age", min_value=0, max_value=120, value=29, step=1, 
                                  help="Age in years")
        
        st.markdown("<br>", unsafe_allow_html=True)
        submit_button = st.form_submit_button(label="Predict")
        
    if submit_button:
        # Create input array
        input_data = pd.DataFrame([[
            pregnancies, glucose, blood_pressure, skin_thickness,
            insulin, bmi, dpf, age
        ]], columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                     'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])
        
        try:
            with st.spinner('Analyzing...'):
                # Scale input
                scaled_data = scaler.transform(input_data)
                
                # Predict
                prediction = model.predict(scaled_data)[0]
                
                # Use predict_proba for confidence if the model supports it
                if hasattr(model, "predict_proba"):
                    prob = model.predict_proba(scaled_data)[0]
                    confidence = prob[prediction] * 100
                    conf_text = f"Confidence Score: {confidence:.2f}%"
                else:
                    conf_text = "Confidence scoring not available."

                # UI Update
                st.divider()
                st.subheader("Prediction Result")
                
                if prediction == 1:
                    st.error("### ⚠️ Diabetic")
                    st.write("The model assessment indicates that the patient **is highly likely to be Diabetic**.")
                    st.info(f"{conf_text}")
                else:
                    st.success("### ✅ Not Diabetic")
                    st.write("The model assessment indicates that the patient **is Not Diabetic**.")
                    st.info(f"{conf_text}")
                
        except Exception as e:
            st.error(f"An unexpected error occurred during prediction: {e}")

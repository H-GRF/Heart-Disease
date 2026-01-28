import streamlit as st
import pandas as pd
import sys
import os

# Add parent directory to path to import from src/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.processing import load_and_preprocess, train_heart_model

st.set_page_config(page_title="M2 Heart AI Project", layout="centered")

@st.cache_resource
def get_resources():
    """Cache the data and model to improve performance."""
    df, encoders = load_and_preprocess('data/heart.csv')
    model = train_heart_model(df)
    return df, encoders, model

df, encoders, model = get_resources()

st.title("💓 Heart Disease Risk Predictor")
st.markdown("Enter the patient's clinical metrics below to estimate risk.")

with st.form("risk_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.slider("Age", 20, 90, 50)
        sex = st.selectbox("Sex", ["M", "F"])
        cp = st.selectbox("Chest Pain Type", ["ASY", "NAP", "ATA", "TA"])
        rbp = st.number_input("Resting Blood Pressure", 80, 200, 120)
        chol = st.number_input("Cholesterol", 0, 600, 200)
        fbs = st.selectbox("Fasting Blood Sugar > 120", [0, 1])
        
    with col2:
        ecg = st.selectbox("Resting ECG", ["Normal", "LVH", "ST"])
        mhr = st.slider("Max Heart Rate", 60, 220, 150)
        exang = st.selectbox("Exercise Angina", ["N", "Y"])
        oldpeak = st.number_input("Oldpeak", 0.0, 6.0, 1.0)
        slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])
    
    submit = st.form_submit_button("Analyze Risk")

if submit:
    # Prepare data for prediction
    input_data = pd.DataFrame([[age, sex, cp, rbp, chol, fbs, ecg, mhr, exang, oldpeak, slope]], 
                             columns=df.columns[:-1])
    
    # Apply encoders
    for col, le in encoders.items():
        input_data[col] = le.transform(input_data[col])
    
    prediction = model.predict_proba(input_data)[0][1]
    
    if prediction > 0.5:
        st.error(f"High Risk Detected: {prediction:.1%}")
    else:
        st.success(f"Low Risk Detected: {prediction:.1%}")
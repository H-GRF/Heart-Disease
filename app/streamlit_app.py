import streamlit as st
import pandas as pd
import sys
import os

# Fix path to allow importing from src folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.processing import load_and_preprocess, train_heart_model

st.set_page_config(page_title="Heart Disease AI Predictor", layout="wide")

@st.cache_resource
def initialize_system():
    # Relative path from the root of the project
    data_path = 'data/heart.csv'
    df, encoders = load_and_preprocess(data_path)
    model = train_heart_model(df)
    return df, encoders, model

df, encoders, model = initialize_system()

st.title("💓 Heart Disease Risk Analysis")
st.write("M2 Software Development Project 2025-2026")

with st.sidebar:
    st.header("Patient Metrics")
    age = st.slider("Age", 20, 90, 50)
    sex = st.selectbox("Sex", ["M", "F"])
    cp = st.selectbox("Chest Pain Type", ["ASY", "NAP", "ATA", "TA"])
    rbp = st.number_input("Resting Blood Pressure", 80, 200, 120)
    chol = st.number_input("Cholesterol", 0, 600, 200)
    fbs = st.selectbox("Fasting Blood Sugar > 120", [0, 1])
    ecg = st.selectbox("Resting ECG", ["Normal", "LVH", "ST"])
    mhr = st.slider("Max Heart Rate", 60, 220, 150)
    exang = st.selectbox("Exercise Angina", ["N", "Y"])
    oldpeak = st.number_input("Oldpeak", 0.0, 6.0, 1.0)
    slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

if st.button("Calculate Heart Attack Risk"):
    # Build input row
    input_data = pd.DataFrame([[age, sex, cp, rbp, chol, fbs, ecg, mhr, exang, oldpeak, slope]], 
                             columns=df.columns[:-1])
    
    # Apply label encoding
    for col, le in encoders.items():
        input_data[col] = le.transform(input_data[col])
    
    risk_prob = model.predict_proba(input_data)[0][1]
    
    if risk_prob > 0.5:
        st.error(f"⚠️ HIGH RISK DETECTED: {risk_prob:.1%}")
    else:
        st.success(f"✅ LOW RISK DETECTED: {risk_prob:.1%}")
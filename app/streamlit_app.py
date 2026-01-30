import streamlit as st
import pandas as pd
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Circle

# Fix path to allow importing from src folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.processing import load_and_preprocess, train_heart_model

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Heart Disease AI Predictor",
    page_icon="💓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# CSS
# -----------------------------
st.markdown(
    """
    <style>
      .block-container { padding-top: 1.0rem; padding-bottom: 2rem; }
      section[data-testid="stSidebar"] { border-right: 1px solid rgba(255,255,255,0.06); }

      .card {
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 18px 18px 14px 18px;
        background: rgba(255,255,255,0.03);
        box-shadow: 0 10px 28px rgba(0,0,0,0.22);
      }

      /* Big CTA button */
      div.stButton > button[kind="primary"] {
        width: 100%;
        padding: 0.85rem 1rem !important;
        font-size: 1.05rem !important;
        font-weight: 800 !important;
        border-radius: 14px !important;
      }

      /* Secondary button */
      div.stButton > button[kind="secondary"] {
        width: 100%;
        border-radius: 12px !important;
        font-weight: 700 !important;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Model init
# -----------------------------
@st.cache_resource
def initialize_system():
    data_path = "data/heart.csv"
    df, encoders = load_and_preprocess(data_path)
    model = train_heart_model(df)
    return df, encoders, model

df, encoders, model = initialize_system()

# -----------------------------
# Defaults + Reset
# -----------------------------
DEFAULTS = {
    "age": 50,
    "sex": "M",
    "cp": "ASY",
    "rbp": 120,
    "chol": 200,
    "fbs": 0,
    "ecg": "Normal",
    "mhr": 150,
    "exang": "N",
    "oldpeak": 1.0,
    "slope": "Up",
}

def reset_inputs():
    for k, v in DEFAULTS.items():
        st.session_state[k] = v
    st.rerun()

# -----------------------------
# Risk bucket
# -----------------------------
def bucket(prob: float):
    if prob >= 0.65:
        return "HIGH RISK", "error"
    elif prob >= 0.35:
        return "MODERATE RISK", "warning"
    return "LOW RISK", "success"

# -----------------------------
# Matplotlib gauge (robust)
# -----------------------------
def plot_gauge(prob: float):
    prob = float(max(0.0, min(1.0, prob)))
    pct = prob * 100

    # Colors by bucket (matplotlib default-ish, no custom theme)
    if prob >= 0.65:
        color = "red"
        label = "HIGH RISK"
    elif prob >= 0.35:
        color = "orange"
        label = "MODERATE RISK"
    else:
        color = "green"
        label = "LOW RISK"

    fig, ax = plt.subplots(figsize=(5.5, 2.8))
    ax.set_aspect("equal")
    ax.axis("off")

    # Background semi-circle
    bg = Wedge((0, 0), 1.0, 0, 180, width=0.18, alpha=0.20)
    ax.add_patch(bg)

    # Filled wedge (0..180)
    angle = 180 * prob
    fill = Wedge((0, 0), 1.0, 0, angle, width=0.18, alpha=0.85)
    fill.set_facecolor(color)
    fill.set_edgecolor(color)
    ax.add_patch(fill)

    # Needle
    theta = np.deg2rad(angle)
    x = 0.85 * np.cos(theta)
    y = 0.85 * np.sin(theta)
    ax.plot([0, x], [0, y], linewidth=4)
    ax.add_patch(Circle((0, 0), 0.04))

    # Text
    ax.text(0, -0.18, f"{pct:.0f}%", ha="center", va="center", fontsize=24, fontweight="bold")
    ax.text(0, -0.35, label, ha="center", va="center", fontsize=11, fontweight="bold", alpha=0.9)

    # Limits (ensure nothing cropped)
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-0.55, 1.15)

    return fig

# -----------------------------
# Header + intro text
# -----------------------------
st.title("💓 Heart Disease Risk Analysis")
st.caption("M2 Software Development Project 2025–2026 • Random Forest")

st.markdown(
    """
**What this app does**  
This application provides an estimated heart disease risk probability based on common clinical variables
(age, chest pain type, blood pressure, cholesterol, ECG results, etc.).
Its purpose is strictly educational: to illustrate how a machine learning model combines medical features
and to highlight which variables have the greatest influence on predictions.

**Important limitations**  
This tool is **not a medical device** and **does not provide a medical diagnosis**.
Predictions depend on the underlying dataset and model assumptions, and may be inaccurate,
especially for inputs outside the training distribution.

If you experience medical symptoms or have health concerns, consult a **qualified healthcare professional**.

In case of an incorrect prediction, misinterpretation, or poor medical decision
resulting from the use of this application, the sole responsibility lies with:

**virgile.PESCE@univ-amu.fr**

"""
)




st.info("Educational demo — not medical advice.", icon="⚠️")
st.markdown("<br>", unsafe_allow_html=True)

# -----------------------------
# Sidebar inputs
# -----------------------------
with st.sidebar:
    st.header("Patient Metrics")
    st.button("↩️ Reset inputs", on_click=reset_inputs, type="secondary")

    age = st.slider(
        "Age (years)",
        18, 100,
        st.session_state.get("age", DEFAULTS["age"]),
        key="age",
        help="Age in years. Risk generally increases with age."
    )

    sex = st.selectbox(
        "Sex",
        ["M", "F"],
        index=["M", "F"].index(st.session_state.get("sex", DEFAULTS["sex"])),
        key="sex",
        help="Dataset uses biological sex recorded in medical documents."
    )

    cp = st.selectbox(
        "Chest Pain Type",
        ["ASY", "ATA", "NAP", "TA"],
        index=["ASY","ATA","NAP","TA"].index(st.session_state.get("cp", DEFAULTS["cp"])),
        key="cp",
        help=(
            "Chest pain type:\n"
            "- ASY: no symptoms\n"
            "- ATA: typical angina\n"
            "- NAP: atypical angina\n"
            "- TA: non-anginal pain"
        )
    )

    rbp = st.number_input(
        "Resting Blood Pressure (mmHg)",
        min_value=80, max_value=220,
        value=int(st.session_state.get("rbp", DEFAULTS["rbp"])),
        step=1,
        key="rbp",
        help="Resting systolic blood pressure (mmHg). Typical ~110–130."
    )

    chol = st.number_input(
        "Cholesterol (mg/dL)",
        min_value=0, max_value=600,
        value=int(st.session_state.get("chol", DEFAULTS["chol"])),
        step=1,
        key="chol",
        help="Total cholesterol (mg/dL). In this dataset, 0 can mean missing."
    )

    fbs = st.selectbox(
        "Fasting Blood Sugar > 120 mg/dL",
        [0, 1],
        index=[0, 1].index(st.session_state.get("fbs", DEFAULTS["fbs"])),
        key="fbs",
        help="1 = Yes (above 120), 0 = No/Unknown."
    )

    ecg = st.selectbox(
        "Resting ECG",
        ["Normal", "LVH", "ST"],
        index=["Normal","LVH","ST"].index(st.session_state.get("ecg", DEFAULTS["ecg"])),
        key="ecg",
        help=(
            "Resting ECG categories:\n"
            "- Normal: no notable abnormality\n"
            "- ST: ST-T wave abnormality (ECG category)\n"
            "- LVH: left ventricular hypertrophy (often linked to hypertension)"
        )
    )

    mhr = st.slider(
        "Max Heart Rate (bpm)",
        60, 220,
        int(st.session_state.get("mhr", DEFAULTS["mhr"])),
        key="mhr",
        help="Maximum heart rate achieved during stress test."
    )

    exang = st.selectbox(
        "Exercise Angina",
        ["N", "Y"],
        index=["N","Y"].index(st.session_state.get("exang", DEFAULTS["exang"])),
        key="exang",
        help="Chest pain/discomfort during physical activity? Y/N."
    )

    oldpeak = st.number_input(
        "Oldpeak",
        min_value=0.0, max_value=6.0,
        value=float(st.session_state.get("oldpeak", DEFAULTS["oldpeak"])),
        step=0.1,
        format="%.1f",
        key="oldpeak",
        help="ST depression induced by exercise (from stress ECG)."
    )

    slope = st.selectbox(
        "ST Slope",
        ["Up", "Flat", "Down"],
        index=["Up","Flat","Down"].index(st.session_state.get("slope", DEFAULTS["slope"])),
        key="slope",
        help="Slope of the ST segment during exercise."
    )

    # Soft warnings (non blocking)
    st.markdown("---")
    st.caption("Input checks (non-blocking)")

    if rbp >= 140:
        st.warning("Resting BP looks high (≥ 140 mmHg).", icon="⚠️")
    elif rbp < 90:
        st.warning("Resting BP looks low (< 90 mmHg).", icon="⚠️")

    if chol == 0:
        st.info("Cholesterol = 0 will be treated as missing (dataset quirk).", icon="ℹ️")
    elif chol >= 240:
        st.warning("Cholesterol looks high (≥ 240 mg/dL).", icon="⚠️")

# -----------------------------
# Build raw input dataframe
# -----------------------------
raw_input = pd.DataFrame(
    [[age, sex, cp, rbp, chol, fbs, ecg, mhr, exang, oldpeak, slope]],
    columns=df.columns[:-1],
)

# -----------------------------
# Layout
# -----------------------------
left, right = st.columns([0.55, 0.45], vertical_alignment="top")

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Current Input Summary")

    summary = pd.DataFrame(
        {
            "Feature": ["Age", "Sex", "ChestPainType", "RestingBP", "Cholesterol", "FastingBS", "RestingECG", "MaxHR", "ExerciseAngina", "Oldpeak", "ST Slope"],
            "Value": [age, sex, cp, rbp, chol, fbs, ecg, mhr, exang, oldpeak, slope],
        }
    )
    st.dataframe(summary, use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Risk Result")

    run_pred = st.button("🧠 Calculate Heart Attack Risk", type="primary")

    if not run_pred:
        st.info("Adjust inputs in the sidebar, then click **Calculate Heart Attack Risk**.", icon="ℹ️")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        # Prepare model input
        input_for_model = raw_input.copy()

        # Chol=0 => median imputation
        if float(input_for_model["Cholesterol"].iloc[0]) == 0.0:
            median_chol = pd.to_numeric(df["Cholesterol"], errors="coerce").replace(0, pd.NA).median()
            if pd.isna(median_chol):
                median_chol = 200
            input_for_model.loc[0, "Cholesterol"] = float(median_chol)

        # Encode categorical columns
        for col, le in encoders.items():
            input_for_model[col] = le.transform(input_for_model[col])

        risk_prob = float(model.predict_proba(input_for_model)[0][1])

        # Matplotlib gauge (always works)
        fig = plot_gauge(risk_prob)
        st.pyplot(fig, use_container_width=True)

        sev, level = bucket(risk_prob)
        if level == "error":
            st.error(f"🔴 {sev} (model output): {risk_prob:.1%}")
        elif level == "warning":
            st.warning(f"🟠 {sev} (model output): {risk_prob:.1%}")
        else:
            st.success(f"🟢 {sev} (model output): {risk_prob:.1%}")

        st.markdown("### Model Insight: Feature Importance")
        try:
            importances = pd.Series(model.feature_importances_, index=input_for_model.columns).sort_values(ascending=False)
            top = importances.head(8).reset_index()
            top.columns = ["Feature", "Importance"]
            st.dataframe(top, use_container_width=True, hide_index=True)
            st.bar_chart(top.set_index("Feature"))
        except Exception as e:
            st.caption(f"Feature importance not available: {e}")

        with st.expander("See model-ready features (encoded)"):
            st.dataframe(input_for_model, use_container_width=True)

        st.caption("Disclaimer: This app is for educational purposes and does not provide medical advice.")
        st.markdown("</div>", unsafe_allow_html=True)
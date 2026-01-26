import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Heart Risk AI", layout="wide")

# Chargement et Préparation du Modèle
@st.cache_resource
def train_model():
    df = pd.read_csv('heart.csv')
    df_ml = df.copy()
    le_dict = {}
    for col in ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']:
        le = LabelEncoder()
        df_ml[col] = le.fit_transform(df_ml[col])
        le_dict[col] = le
    
    X = df_ml.drop('HeartDisease', axis=1)
    y = df_ml['HeartDisease']
    model = RandomForestClassifier(random_state=42).fit(X, y)
    return model, le_dict, df

model, encoders, df = train_model()

# Interface Streamlit
st.title("💓 Diagnostic & Analyse Cardiaque")

tab1, tab2 = st.tabs(["🔍 Explorateur de Données", "⚖️ Évaluation de Risque"])

with tab1:
    st.sidebar.header("Filtres")
    gender = st.sidebar.multiselect("Sexe", df["Sex"].unique(), df["Sex"].unique())
    age_range = st.sidebar.slider("Tranche d'âge", int(df.Age.min()), int(df.Age.max()), (20, 80))
    
    filtered_df = df[(df.Sex.isin(gender)) & (df.Age.between(age_range[0], age_range[1]))]
    
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"Moyenne Cholestérol: {filtered_df['Cholesterol'].mean():.1f}")
        fig, ax = plt.subplots()
        sns.histplot(filtered_df, x="MaxHR", hue="HeartDisease", kde=True, ax=ax)
        st.pyplot(fig)
    with col2:
        st.write(f"Nombre de patients filtrés: {len(filtered_df)}")
        fig, ax = plt.subplots()
        sns.boxplot(data=filtered_df, x="HeartDisease", y="Age", ax=ax)
        st.pyplot(fig)

with tab2:
    st.header("Entrez vos informations")
    c1, c2, c3 = st.columns(3)
    
    age = c1.number_input("Âge", 1, 100, 40)
    sex = c2.selectbox("Sexe", ["M", "F"])
    cp = c3.selectbox("Type Douleur (ChestPain)", ["ATA", "NAP", "ASY", "TA"])
    rbp = c1.number_input("Pression artérielle (RestingBP)", 50, 200, 120)
    chol = c2.number_input("Cholestérol", 0, 600, 200)
    fbs = c3.selectbox("Sucre à jeun > 120 mg/dl", [0, 1])
    ecg = c1.selectbox("ECG au repos", ["Normal", "ST", "LVH"])
    mhr = c2.number_input("Fréquence Cardiaque Max (MaxHR)", 50, 210, 150)
    exang = c3.selectbox("Angine d'effort (ExerciseAngina)", ["N", "Y"])
    oldpeak = c1.number_input("Oldpeak (Dépression ST)", 0.0, 10.0, 0.0)
    slope = c2.selectbox("Pente ST (ST_Slope)", ["Up", "Flat", "Down"])

    if st.button("Calculer mon risque"):
        # Transformation des entrées
        input_data = pd.DataFrame([[age, sex, cp, rbp, chol, fbs, ecg, mhr, exang, oldpeak, slope]],
                                 columns=df.columns[:-1])
        for col, le in encoders.items():
            input_data[col] = le.transform(input_data[col])
        
        prob = model.predict_proba(input_data)[0][1]
        
        if prob > 0.5:
            st.error(f"⚠️ Risque Élevé : {prob:.1%}")
            st.write("Veuillez consulter un spécialiste.")
        else:
            st.success(f"✅ Risque Faible : {prob:.1%}")
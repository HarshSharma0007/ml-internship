import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and preprocessor
model = joblib.load("../outputs/model_rf.pkl")
preprocessor = joblib.load("../outputs/preprocessor.pkl")

st.set_page_config(page_title="Heart Disease Predictor", layout="centered")
st.title("🫀 Heart Disease Prediction App")
st.markdown("Enter patient vitals to predict heart disease risk.")

# Input fields
age = st.slider("Age", 29, 77, 54)
sex = st.selectbox("Sex", ["Female", "Male"])
cp = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"])
trestbps = st.slider("Resting Blood Pressure (mm Hg)", 94, 200, 130)
chol = st.slider("Serum Cholesterol (mg/dL)", 126, 564, 245)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", ["No", "Yes"])
restecg = st.selectbox("Resting ECG", ["Normal", "ST-T Abnormality", "LV Hypertrophy"])
thalach = st.slider("Max Heart Rate Achieved", 71, 202, 150)
exang = st.selectbox("Exercise-Induced Angina", ["No", "Yes"])
oldpeak = st.slider("ST Depression (Oldpeak)", 0.0, 6.2, 1.0)
slope = st.selectbox("Slope of ST Segment", ["Upward", "Flat", "Downward"])

# Map inputs to model format
input_dict = {
    "age": age,
    "sex": 1 if sex == "Male" else 0,
    "cp": ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"].index(cp) + 1,
    "trestbps": trestbps,
    "chol": chol,
    "fbs": 1 if fbs == "Yes" else 0,
    "restecg": ["Normal", "ST-T Abnormality", "LV Hypertrophy"].index(restecg),
    "thalach": thalach,
    "exang": 1 if exang == "Yes" else 0,
    "oldpeak": oldpeak,
    "slope": ["Upward", "Flat", "Downward"].index(slope) + 1
}

input_df = pd.DataFrame([input_dict])

# Preprocess and predict
X_processed = preprocessor.transform(input_df)
prediction = model.predict(X_processed)[0]
proba = model.predict_proba(X_processed)[0][1]

# Output
if st.button("Predict"):
    if prediction == 1:
        st.error(f"⚠️ High risk of heart disease. Probability: {proba:.2f}")
    else:
        st.success(f"✅ No heart disease detected. Probability: {proba:.2f}")

import streamlit as st
import pandas as pd
import numpy as np
import shap
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")
st.title("🛡️ Fraud Detection Dashboard")

# --- Load Model ---
model = joblib.load("models/fraud_model.pkl")

# --- Load Data ---
@st.cache_data
def load_data():
    df = pd.read_pickle("data/processed/df_with_terminal_fraud.pkl")
    X = df.drop(columns=["TX_FRAUD"])
    y = df["TX_FRAUD"]
    X_train, _, _, _ = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    return X_train.select_dtypes(include="number").astype("float64")

X_train = load_data()

# --- Sidebar Inputs ---
st.sidebar.header("🧪 Simulate Transaction")
amount = st.sidebar.slider("Transaction Amount", 0, 10000, 500)
amount_dev = st.sidebar.slider("Amount Deviation", 0.0, 1.0, 0.2)
terminal_fraud = st.sidebar.number_input("Terminal Fraud Count (28d)", 0, 50, 2)
customer_avg = st.sidebar.slider("Customer Avg Amount (14d)", 0, 10000, 300)
terminal_avg = st.sidebar.slider("Terminal Avg Amount", 0, 10000, 400)
hour = st.sidebar.slider("Transaction Hour", 0, 23, 14)
day = st.sidebar.slider("Transaction Day", 0, 6, 2)
weekday = day % 7
is_high_amount = int(amount > 5000)

# --- Create Input DataFrame ---
# --- Create Input DataFrame ---
input_df = pd.DataFrame({
    "TX_AMOUNT": [amount],
    "is_high_amount": [is_high_amount],
    "terminal_fraud_count_28d": [terminal_fraud],
    "customer_avg_amount_14d": [customer_avg],
    "amount_deviation": [amount_dev],
    "TX_HOUR": [hour],
    "TX_WEEKDAY": [weekday]
})

# Ensure correct column order and no missing features
expected_features = list(model.feature_names_in_)
missing = set(expected_features) - set(input_df.columns)
extra = set(input_df.columns) - set(expected_features)

if missing or extra:
    st.error(f"Feature mismatch! Missing: {missing}, Extra: {extra}")
    st.stop()

input_df = input_df[expected_features]


# --- Prediction ---
prediction = model.predict(input_df)[0]
probability = model.predict_proba(input_df)[0][1]

st.subheader("🔍 Prediction Result")
st.metric("Fraud Probability", f"{probability:.2%}")
if prediction == 0:
    st.success("✅ Legitimate Transaction")
else:
    st.error("⚠️ Fraud Detected")


# --- SHAP Explanation ---

st.subheader("📊 SHAP Explanation")
explainer = shap.Explainer(model, X_train)
shap_values = explainer(input_df, check_additivity=False)


# Force Plot
shap.force_plot(
    explainer.expected_value[0],
    shap_values.values[0][..., 0],
    input_df,
    matplotlib=True,
    show=False
)

fig_force = plt.gcf()  # ✅ Capture the current figure
st.subheader("Force Plot")
st.pyplot(fig_force)




# Decision Plot


# Create a new figure
fig = plt.figure(figsize=(10, 3))

# Generate decision plot on this figure
shap.decision_plot(
    explainer.expected_value[0],
    shap_values.values[0][..., 0],
    feature_names=input_df.columns.tolist(),
    show=False  # Prevent auto-rendering
)

# Render in Streamlit
st.subheader("Decision Plot")
st.pyplot(fig)  # ✅ Explicit figure passed — no warning




# --- Footer ---
st.markdown("---")
st.caption("Built by Harsh | Modular, Explainable, Reproducible ML")

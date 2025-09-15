# 🛡️ Project 5: Fraud Transaction Detection

## 📌 Overview

This project builds a robust, interpretable machine learning pipeline to detect fraudulent financial transactions. It combines modular feature engineering, explainable modeling, and an interactive Streamlit dashboard to simulate and visualize fraud risk in real time.

---

## 📂 Project Structure

```bash
Project_5_Fraud_Transaction_Detection/
│
├── data/
│   └── processed/
│       └── df_with_terminal_fraud.pkl
│
├── models/
│   └── fraud_model.pkl
├── notebooks/
│   ├── 01_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_training.ipynb
│
├── app.py
├── src/
│   ├── config.py
│   ├── feature_engineering.py
│   ├── load_data.py
│   └── train_model.py
│
├── README.md
└── requirements.txt
```

---

## 🧠 Problem Statement

Online financial fraud is rising rapidly. Detecting fraudulent transactions in real time is critical for financial institutions. This project aims to:

- Identify key behavioral and transactional indicators of fraud
- Build a high-performance classifier with interpretable outputs
- Deploy a user-facing app for simulation and explanation

---

## 📊 Dataset

The dataset includes:

- `TX_AMOUNT`: Transaction amount
- `TX_HOUR`, `TX_WEEKDAY`: Temporal features
- `customer_avg_amount_14d`: Rolling customer behavior
- `terminal_fraud_count_28d`: Terminal-level fraud history
- `amount_deviation`: Normalized deviation from expected behavior
- `TX_FRAUD`: Binary target label

> Data is preprocessed and stored as `df_with_terminal_fraud.pkl`.

---

## 🧪 Feature Engineering

Key transformations include:

- Rolling averages and fraud counts over time windows
- Binary flags for high-risk behavior (`is_high_amount`)
- Normalized deviation metrics (`amount_deviation`)
- Temporal encoding (`TX_WEEKDAY`, `TX_HOUR`)

All transformations are modularized in `utils/feature_engineering.py`.

---

## 🤖 Model Pipeline

- **Algorithm**: Gradient Boosting Classifier (or Random Forest)
- **Handling Imbalance**: Stratified sampling, recall optimization
- **Explainability**: SHAP TreeExplainer with visual diagnostics
- **Persistence**: Model saved as `fraud_model.pkl`

Model training and evaluation logic lives in `utils/model_utils.py`.

---

## 🧵 Streamlit App

Launch with:

```bash
streamlit run app.py
```

Features:

- Sidebar simulation of transaction parameters
- Real-time fraud prediction with probability
- SHAP force and decision plots for interpretability
- Input validation and feature alignment checks

---

## 📈 Evaluation Metrics

- Precision, Recall, F1-score
- Confusion Matrix
- SHAP-based feature importance
- Decision plots for individual predictions

---

## 🧰 Requirements

```txt
streamlit
pandas
numpy
scikit-learn
shap
matplotlib
joblib
```

Install with:

```bash
pip install -r requirements.txt
```

---

## 🚀 Future Enhancements

- Batch prediction via CSV upload
- MLflow integration for model tracking
- Deployment on Streamlit Cloud or Hugging Face Spaces
- Real-time fraud alerting via API

---

## 👨‍💻 Author

**Harsh Sharma**  
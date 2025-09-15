# 🫀 Heart Disease Prediction Project

## 🎯 Objective

Build a high-performance, interpretable machine learning model to predict the presence of heart disease using patient vitals and clinical indicators. The goal is to create a modular, reproducible pipeline suitable for educational demos, clinical reasoning, and deployment.

---

## 📦 Dataset Overview

The dataset contains anonymized patient records with 13 features and a binary target (`0 = Normal`, `1 = Heart Disease`). Features include age, sex, chest pain type, cholesterol levels, ECG results, and more.

| Feature      | Description |
|--------------|-------------|
| `age`        | Age in years |
| `sex`        | 0 = female, 1 = male |
| `cp`         | Chest pain type (1–4) |
| `trestbps`   | Resting blood pressure (mm Hg) |
| `chol`       | Serum cholesterol (mg/dL) |
| `fbs`        | Fasting blood sugar > 120 mg/dL (1 = true; 0 = false) |
| `restecg`    | Resting ECG results (0–2) |
| `thalach`    | Maximum heart rate achieved |
| `exang`      | Exercise-induced angina (1 = yes; 0 = no) |
| `oldpeak`    | ST depression induced by exercise |
| `slope`      | Slope of peak exercise ST segment (1–3) |
| `target`     | 0 = Normal, 1 = Heart Disease |

---

## 🧪 Workflow Summary

### 1. **Exploratory Data Analysis (EDA)**
- Visualized distributions, outliers, and correlations
- Analyzed class balance and feature-target relationships
- Identified clinically relevant patterns (e.g., age vs. max heart rate)

### 2. **Preprocessing**
- Scaled numeric features using `StandardScaler`
- One-hot encoded nominal categorical features
- Preserved binary features
- Applied SMOTE to balance classes

### 3. **Modeling**
- Baseline: Logistic Regression
- Final Model: Random Forest with threshold tuning
- Evaluation metrics: Accuracy, Precision, Recall, F1-score, ROC AUC

### 4. **Optimization**
- Tuned decision threshold to reduce false negatives
- Achieved near-perfect recall for heart disease cases
- Visualized feature importance

### 5. **Deployment**
- Built a Streamlit app for live predictions
- Integrated model and preprocessing pipeline
- User-friendly interface with clinical terminology

---

## 📊 Final Model Performance (Random Forest + SMOTE + Threshold = 0.4)

| Metric        | Value |
|---------------|-------|
| **Accuracy**  | 0.95  |
| **Precision** | 0.94  |
| **Recall**    | 0.98  |
| **F1 Score**  | 0.96  |
| **ROC AUC**   | 0.97  |
| **False Negatives** | 3 |

📌 *Interpretation*: The model correctly identifies 98% of heart disease cases, with only 3 false negatives out of 238 test samples—clinically robust and ethically sound.

---

## 🚀 Streamlit App

A demo-ready web app allows users to input patient vitals and receive instant predictions. The app includes:
- Intuitive sliders and dropdowns
- Real-time probability output
- Clinical language for accessibility

To run locally:
```bash
streamlit run app.py

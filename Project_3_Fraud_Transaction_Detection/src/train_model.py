import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

def train_model_with_mlflow(df, feature_cols, target_col='TX_FRAUD'):
    """
    Trains a Random Forest model with MLflow tracking.
    Logs parameters, metrics, model, and input example.
    """
    X = df[feature_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )

    with mlflow.start_run():
        # Define model
        model = RandomForestClassifier(
            n_estimators=100,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )

        # Train
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        # Metrics
        roc_auc = roc_auc_score(y_test, y_proba)
        report = classification_report(y_test, y_pred, output_dict=True)

        # Log parameters
        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("class_weight", "balanced")

        # Log metrics
        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.log_metric("precision_fraud", report['1']['precision'])
        mlflow.log_metric("recall_fraud", report['1']['recall'])
        mlflow.log_metric("f1_fraud", report['1']['f1-score'])

        # Log model with input example
        input_example = X_train.iloc[:5]
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="random_forest_model",
            input_example=input_example,
            registered_model_name="random_forest_model"
        )

        # Print report
        print(classification_report(y_test, y_pred))
        print("ROC AUC:", roc_auc)

    return model, X_test, y_test


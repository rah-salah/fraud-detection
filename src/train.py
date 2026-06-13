import os
import logging
import joblib
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    classification_report,
    f1_score,
    roc_auc_score,
    average_precision_score
)
from src.data_processing import (
    load_data,
    clean_fraud_data,
    clean_credit_data,
    merge_geolocation,
    engineer_features,
    transform_features,
    handle_imbalance
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    metrics = {
        "f1": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_test, y_prob), 4),
        "auc_pr": round(average_precision_score(y_test, y_prob), 4),
    }
    logger.info(f"Model: {model_name}")
    logger.info(f"F1: {metrics['f1']}")
    logger.info(f"ROC-AUC: {metrics['roc_auc']}")
    logger.info(f"AUC-PR: {metrics['auc_pr']}")
    logger.info(classification_report(y_test, y_pred))
    return metrics


def train_fraud_models(fraud_df, ip_df):
    logger.info("Training models on Fraud_Data...")
    fraud_df = clean_fraud_data(fraud_df)
    fraud_df = merge_geolocation(fraud_df, ip_df)
    fraud_df = engineer_features(fraud_df)
    fraud_df = transform_features(fraud_df, "class")
    X = fraud_df.drop(columns=["class"])
    y = fraud_df["class"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, y_train = handle_imbalance(X_train, y_train)
    os.makedirs("models", exist_ok=True)
    mlflow.set_experiment("fraud_ecommerce")
    with mlflow.start_run(run_name="LR_Fraud"):
        lr = LogisticRegression(
            class_weight="balanced",
            random_state=42,
            max_iter=1000
        )
        lr.fit(X_train, y_train)
        metrics = evaluate_model(lr, X_test, y_test, "LR_Fraud")
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(lr, "lr_fraud")
        joblib.dump(lr, "models/lr_fraud.pkl")
    with mlflow.start_run(run_name="RF_Fraud"):
        rf = GridSearchCV(
            RandomForestClassifier(
                class_weight="balanced",
                random_state=42,
                n_jobs=-1
            ),
            {"n_estimators": [100, 200], "max_depth": [5, 10, None]},
            cv=3, scoring="average_precision", n_jobs=-1
        )
        rf.fit(X_train, y_train)
        best_rf = rf.best_estimator_
        metrics = evaluate_model(best_rf, X_test, y_test, "RF_Fraud")
        mlflow.log_params(rf.best_params_)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(best_rf, "rf_fraud")
        joblib.dump(best_rf, "models/rf_fraud.pkl")
    logger.info("Fraud models done!")


def train_credit_models(credit_df):
    logger.info("Training models on CreditCard...")
    credit_df = clean_credit_data(credit_df)
    X = credit_df.drop(columns=["Class"])
    y = credit_df["Class"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, y_train = handle_imbalance(X_train, y_train)
    os.makedirs("models", exist_ok=True)
    mlflow.set_experiment("fraud_creditcard")
    with mlflow.start_run(run_name="LR_Credit"):
        lr = LogisticRegression(
            class_weight="balanced",
            random_state=42,
            max_iter=1000
        )
        lr.fit(X_train, y_train)
        metrics = evaluate_model(lr, X_test, y_test, "LR_Credit")
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(lr, "lr_credit")
        joblib.dump(lr, "models/lr_credit.pkl")
    with mlflow.start_run(run_name="RF_Credit"):
        rf = GridSearchCV(
            RandomForestClassifier(
                class_weight="balanced",
                random_state=42,
                n_jobs=-1
            ),
            {"n_estimators": [100, 200], "max_depth": [5, 10, None]},
            cv=3, scoring="average_precision", n_jobs=-1
        )
        rf.fit(X_train, y_train)
        best_rf = rf.best_estimator_
        metrics = evaluate_model(best_rf, X_test, y_test, "RF_Credit")
        mlflow.log_params(rf.best_params_)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(best_rf, "rf_credit")
        joblib.dump(best_rf, "models/rf_credit.pkl")
    logger.info("Credit models done!")


if __name__ == "__main__":
    fraud_df, ip_df, credit_df = load_data()
    train_fraud_models(fraud_df, ip_df)
    train_credit_models(credit_df)
    logger.info("All models trained and saved!")

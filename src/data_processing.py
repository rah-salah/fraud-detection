import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_data():
    """Load all three datasets."""
    logger.info("Loading datasets...")
    fraud_df = pd.read_csv("data/raw/Fraud_Data.csv")
    ip_df = pd.read_csv("data/raw/IpAddress_to_Country.csv")
    credit_df = pd.read_csv("data/raw/creditcard.csv")
    logger.info(f"Fraud_Data: {fraud_df.shape}")
    logger.info(f"IpAddress: {ip_df.shape}")
    logger.info(f"CreditCard: {credit_df.shape}")
    return fraud_df, ip_df, credit_df


def clean_fraud_data(df):
    """Clean Fraud_Data.csv."""
    logger.info("Cleaning Fraud_Data...")
    # Remove duplicates
    df = df.drop_duplicates()
    # Fix data types
    df["signup_time"] = pd.to_datetime(df["signup_time"])
    df["purchase_time"] = pd.to_datetime(df["purchase_time"])
    # Handle missing values
    df = df.dropna()
    logger.info(f"After cleaning: {df.shape}")
    return df


def clean_credit_data(df):
    """Clean creditcard.csv."""
    logger.info("Cleaning CreditCard data...")
    df = df.drop_duplicates()
    df = df.dropna()
    logger.info(f"After cleaning: {df.shape}")
    return df


def ip_to_int(ip_str):
    """Convert IP address string to integer."""
    try:
        parts = str(ip_str).split(".")
        if len(parts) != 4:
            return 0
        return (int(parts[0]) * 16777216 +
                int(parts[1]) * 65536 +
                int(parts[2]) * 256 +
                int(parts[3]))
    except Exception:
        return 0


def merge_geolocation(fraud_df, ip_df):
    """Merge fraud data with IP country data."""
    logger.info("Merging geolocation data...")
    # Convert IPs to integers
    fraud_df["ip_int"] = fraud_df["ip_address"].apply(ip_to_int)
    ip_df["lower_int"] = ip_df["lower_bound_ip_address"].apply(ip_to_int)
    ip_df["upper_int"] = ip_df["upper_bound_ip_address"].apply(ip_to_int)
    # Sort for merge_asof
    fraud_df = fraud_df.sort_values("ip_int")
    ip_df = ip_df.sort_values("lower_int")
    # Range-based merge
    merged = pd.merge_asof(
        fraud_df,
        ip_df[["lower_int", "upper_int", "country"]],
        left_on="ip_int",
        right_on="lower_int",
        direction="backward"
    )
    # Keep only valid matches
    merged["country"] = merged.apply(
        lambda x: x["country"]
        if x["ip_int"] <= x["upper_int"]
        else "Unknown", axis=1
    )
    merged["country"] = merged["country"].fillna("Unknown")
    logger.info(f"After merge: {merged.shape}")
    return merged


def engineer_features(df):
    """Create new features for fraud detection."""
    logger.info("Engineering features...")
    # Time since signup
    df["time_since_signup"] = (
        df["purchase_time"] - df["signup_time"]
    ).dt.total_seconds() / 3600
    # Hour and day features
    df["hour_of_day"] = df["purchase_time"].dt.hour
    df["day_of_week"] = df["purchase_time"].dt.dayofweek
    # Transaction velocity per user
    df = df.sort_values("purchase_time")
    df["transaction_count"] = df.groupby("user_id").cumcount() + 1
    logger.info(f"Features created: {df.shape}")
    return df


def transform_features(df, target_col):
    """Scale and encode features."""
    logger.info("Transforming features...")
    # One-hot encode categoricals
    cat_cols = ["source", "browser", "sex", "country"]
    cat_cols = [c for c in cat_cols if c in df.columns]
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    # Drop non-feature columns
    drop_cols = [
        "user_id", "device_id", "ip_address",
        "signup_time", "purchase_time", "ip_int",
        "lower_int", "upper_int"
    ]
    drop_cols = [c for c in drop_cols if c in df.columns]
    df = df.drop(columns=drop_cols)
    # Scale numerical features
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    num_cols = [c for c in num_cols if c != target_col]
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])
    logger.info(f"After transformation: {df.shape}")
    return df


def handle_imbalance(X_train, y_train):
    """Apply SMOTE to training data only."""
    logger.info("Handling class imbalance with SMOTE...")
    logger.info(f"Before SMOTE: {y_train.value_counts().to_dict()}")
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    logger.info(f"After SMOTE: {pd.Series(y_resampled).value_counts().to_dict()}")
    return X_resampled, y_resampled


if __name__ == "__main__":
    fraud_df, ip_df, credit_df = load_data()
    fraud_df = clean_fraud_data(fraud_df)
    credit_df = clean_credit_data(credit_df)
    fraud_df = merge_geolocation(fraud_df, ip_df)
    fraud_df = engineer_features(fraud_df)
    fraud_df = transform_features(fraud_df, "class")
    X = fraud_df.drop(columns=["class"])
    y = fraud_df["class"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, y_train = handle_imbalance(X_train, y_train)
    import os
    os.makedirs("data/processed", exist_ok=True)
    fraud_df.to_csv("data/processed/fraud_processed.csv", index=False)
    credit_df.to_csv("data/processed/credit_processed.csv", index=False)
    logger.info("All preprocessing done!")

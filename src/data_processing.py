import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_data(fraud_path="data/raw/Fraud_Data.csv",
              ip_path="data/raw/IpAddress_to_Country.csv",
              credit_path="data/raw/creditcard.csv"):
    """
    Load all three raw datasets from CSV files.

    Args:
        fraud_path (str): Path to Fraud_Data.csv
        ip_path (str): Path to IpAddress_to_Country.csv
        credit_path (str): Path to creditcard.csv

    Returns:
        tuple: (fraud_df, ip_df, credit_df) as DataFrames

    Raises:
        FileNotFoundError: If any of the CSV files do not exist
    """
    for path in [fraud_path, ip_path, credit_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Data file not found: {path}")

    logger.info("Loading datasets...")
    fraud_df = pd.read_csv(fraud_path)
    ip_df = pd.read_csv(ip_path)
    credit_df = pd.read_csv(credit_path)

    logger.info(f"Fraud_Data: {fraud_df.shape}")
    logger.info(f"IpAddress: {ip_df.shape}")
    logger.info(f"CreditCard: {credit_df.shape}")
    return fraud_df, ip_df, credit_df


def clean_fraud_data(df):
    """
    Clean the Fraud_Data DataFrame.

    Steps:
        - Remove duplicate rows
        - Parse signup_time and purchase_time to datetime
        - Drop rows with missing values

    Args:
        df (pd.DataFrame): Raw fraud DataFrame

    Returns:
        pd.DataFrame: Cleaned DataFrame

    Raises:
        ValueError: If required columns are missing
    """
    required_cols = ["signup_time", "purchase_time", "user_id", "ip_address", "class"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in fraud data: {missing}")

    logger.info("Cleaning Fraud_Data...")
    df = df.drop_duplicates().copy()
    df["signup_time"] = pd.to_datetime(df["signup_time"])
    df["purchase_time"] = pd.to_datetime(df["purchase_time"])
    df = df.dropna()
    logger.info(f"After cleaning: {df.shape}")
    return df


def clean_credit_data(df):
    """
    Clean the creditcard DataFrame.

    Steps:
        - Remove duplicate rows
        - Drop rows with missing values

    Args:
        df (pd.DataFrame): Raw credit card DataFrame

    Returns:
        pd.DataFrame: Cleaned DataFrame

    Raises:
        ValueError: If required columns are missing
    """
    required_cols = ["Amount", "Class"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in credit data: {missing}")

    logger.info("Cleaning CreditCard data...")
    df = df.drop_duplicates()
    df = df.dropna()
    logger.info(f"After cleaning: {df.shape}")
    return df


def ip_to_int(ip_str):
    """
    Convert an IP address string to an integer for range-based lookup.

    Example:
        '192.168.1.1' -> 3232235777

    Args:
        ip_str (str): IP address as a dotted string

    Returns:
        int: Integer representation, or 0 if conversion fails
    """
    try:
        parts = str(ip_str).split(".")
        if len(parts) != 4:
            return 0
        result = int(parts[0]) * 16777216
        result += int(parts[1]) * 65536
        result += int(parts[2]) * 256
        result += int(parts[3])
        return result
    except Exception:
        return 0


def merge_geolocation(fraud_df, ip_df):
    """
    Merge fraud transactions with country data using IP address ranges.

    Converts IP addresses to integers, then uses an asof merge to match
    each transaction IP to the correct country range. Unmatched IPs are
    labeled 'Unknown'.

    Args:
        fraud_df (pd.DataFrame): Cleaned fraud DataFrame with ip_address column
        ip_df (pd.DataFrame): IP-to-country lookup DataFrame

    Returns:
        pd.DataFrame: fraud_df with a new 'country' column

    Raises:
        ValueError: If required columns are missing from either DataFrame
    """
    if "ip_address" not in fraud_df.columns:
        raise ValueError("fraud_df must contain 'ip_address' column")
    required_ip_cols = ["lower_bound_ip_address", "upper_bound_ip_address", "country"]
    missing = [c for c in required_ip_cols if c not in ip_df.columns]
    if missing:
        raise ValueError(f"ip_df missing columns: {missing}")

    logger.info("Merging geolocation data...")
    fraud_df = fraud_df.copy()
    ip_df = ip_df.copy()

    fraud_df["ip_int"] = fraud_df["ip_address"].apply(ip_to_int)
    ip_df["lower_int"] = ip_df["lower_bound_ip_address"].astype("int64")
    ip_df["upper_int"] = ip_df["upper_bound_ip_address"].astype("int64")

    fraud_df = fraud_df.sort_values("ip_int")
    ip_df = ip_df.sort_values("lower_int")

    merged = pd.merge_asof(
        fraud_df,
        ip_df[["lower_int", "upper_int", "country"]],
        left_on="ip_int",
        right_on="lower_int",
        direction="backward"
    )

    merged["country"] = merged.apply(
        lambda x: x["country"] if pd.notna(x.get("upper_int"))
        and x["ip_int"] <= x["upper_int"] else "Unknown",
        axis=1
    )
    merged["country"] = merged["country"].fillna("Unknown")
    logger.info(f"After merge: {merged.shape}")
    logger.info(f"Unknown countries: {(merged['country'] == 'Unknown').sum()}")
    return merged


def engineer_features(df):
    """
    Create new features from raw fraud transaction data.

    New features:
        - time_since_signup: Hours between signup and purchase
        - hour_of_day: Hour of the purchase (0-23)
        - day_of_week: Day of week (0=Monday, 6=Sunday)
        - transaction_count: Cumulative transaction count per user (velocity)

    Args:
        df (pd.DataFrame): Cleaned and geo-merged fraud DataFrame

    Returns:
        pd.DataFrame: DataFrame with new feature columns added

    Raises:
        ValueError: If required datetime columns are missing
    """
    required_cols = ["signup_time", "purchase_time", "user_id"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for feature engineering: {missing}")

    logger.info("Engineering features...")
    df = df.copy()
    df["time_since_signup"] = (
        df["purchase_time"] - df["signup_time"]
    ).dt.total_seconds() / 3600
    df["hour_of_day"] = df["purchase_time"].dt.hour
    df["day_of_week"] = df["purchase_time"].dt.dayofweek
    df = df.sort_values("purchase_time")
    df["transaction_count"] = df.groupby("user_id").cumcount() + 1
    logger.info(f"Features created: {df.shape}")
    return df


def transform_features(df, target_col):
    """
    Apply one-hot encoding to categorical columns and scale numerical features.

    Steps:
        - One-hot encode: source, browser, sex, country
        - Drop identifier and datetime columns
        - StandardScale all remaining numerical columns

    Args:
        df (pd.DataFrame): Feature-engineered DataFrame
        target_col (str): Name of the target column (excluded from scaling)

    Returns:
        pd.DataFrame: Transformed DataFrame ready for modeling

    Raises:
        ValueError: If target column is not found in DataFrame
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame")

    logger.info("Transforming features...")
    df = df.copy()

    cat_cols = ["source", "browser", "sex", "country"]
    cat_cols = [c for c in cat_cols if c in df.columns]
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    drop_cols = [
        "user_id", "device_id", "ip_address",
        "signup_time", "purchase_time", "ip_int",
        "lower_int", "upper_int"
    ]
    drop_cols = [c for c in drop_cols if c in df.columns]
    df = df.drop(columns=drop_cols)

    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    num_cols = [c for c in num_cols if c != target_col]

    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])
    logger.info(f"After transformation: {df.shape}")
    return df


def handle_imbalance(X_train, y_train):
    """
    Apply SMOTE to balance the training data.

    IMPORTANT: Only call this on training data AFTER the train/test split.
    Never apply SMOTE before splitting — it causes data leakage.

    Args:
        X_train (pd.DataFrame or np.ndarray): Training features
        y_train (pd.Series or np.ndarray): Training labels

    Returns:
        tuple: (X_resampled, y_resampled) with balanced classes

    Raises:
        ValueError: If y_train contains only one class
    """
    if len(np.unique(y_train)) < 2:
        raise ValueError("y_train must contain at least two classes for SMOTE")

    logger.info("Applying SMOTE to training data only...")
    logger.info(f"Before SMOTE: {pd.Series(y_train).value_counts().to_dict()}")
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    logger.info(f"After SMOTE: {pd.Series(y_res).value_counts().to_dict()}")
    return X_res, y_res


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

    os.makedirs("data/processed", exist_ok=True)
    fraud_df.to_csv("data/processed/fraud_processed.csv", index=False)
    credit_df.to_csv("data/processed/credit_processed.csv", index=False)
    logger.info("All preprocessing done!")
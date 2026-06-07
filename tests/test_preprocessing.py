import sys
import os
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_processing import (
    ip_to_int,
    clean_fraud_data,
    clean_credit_data,
    engineer_features,
    transform_features,
    handle_imbalance,
    merge_geolocation,
)


# ─── ip_to_int ────────────────────────────────────────────────

def test_ip_to_int_standard():
    assert ip_to_int("192.168.1.1") == 3232235777

def test_ip_to_int_zeros():
    assert ip_to_int("0.0.0.0") == 0

def test_ip_to_int_known():
    assert ip_to_int("1.0.0.0") == 16777216

def test_ip_to_int_max():
    assert ip_to_int("255.255.255.255") == 4294967295

def test_ip_to_int_invalid_string():
    assert ip_to_int("invalid") == 0

def test_ip_to_int_empty_string():
    assert ip_to_int("") == 0

def test_ip_to_int_nan():
    assert ip_to_int(float("nan")) == 0

def test_ip_to_int_wrong_segments():
    assert ip_to_int("192.168.1") == 0


# ─── clean_fraud_data ─────────────────────────────────────────

def make_fraud_df():
    """Helper: minimal valid fraud DataFrame."""
    return pd.DataFrame({
        "user_id": [1, 2, 3],
        "signup_time": ["2023-01-01 10:00:00", "2023-01-02 11:00:00", "2023-01-03 12:00:00"],
        "purchase_time": ["2023-01-02 10:00:00", "2023-01-03 11:00:00", "2023-01-04 12:00:00"],
        "purchase_value": [100, 200, 300],
        "ip_address": ["192.168.1.1", "10.0.0.1", "172.16.0.1"],
        "source": ["SEO", "Ads", "Direct"],
        "browser": ["Chrome", "Firefox", "Safari"],
        "sex": ["M", "F", "M"],
        "age": [25, 30, 35],
        "class": [0, 1, 0],
        "device_id": ["d1", "d2", "d3"],
    })

def test_clean_fraud_data_removes_duplicates():
    df = make_fraud_df()
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    result = clean_fraud_data(df)
    assert len(result) == 3

def test_clean_fraud_data_parses_dates():
    df = make_fraud_df()
    result = clean_fraud_data(df)
    assert pd.api.types.is_datetime64_any_dtype(result["signup_time"])
    assert pd.api.types.is_datetime64_any_dtype(result["purchase_time"])

def test_clean_fraud_data_drops_nulls():
    df = make_fraud_df()
    df.loc[0, "purchase_value"] = None
    result = clean_fraud_data(df)
    assert len(result) == 2

def test_clean_fraud_data_missing_columns():
    df = pd.DataFrame({"user_id": [1], "class": [0]})
    with pytest.raises(ValueError, match="Missing required columns"):
        clean_fraud_data(df)


# ─── clean_credit_data ────────────────────────────────────────

def make_credit_df():
    """Helper: minimal valid credit card DataFrame."""
    return pd.DataFrame({
        "Time": [0, 1, 2],
        "V1": [1.0, -1.0, 0.5],
        "Amount": [100.0, 50.0, 200.0],
        "Class": [0, 1, 0],
    })

def test_clean_credit_data_removes_duplicates():
    df = make_credit_df()
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    result = clean_credit_data(df)
    assert len(result) == 3

def test_clean_credit_data_drops_nulls():
    df = make_credit_df()
    df.loc[0, "Amount"] = None
    result = clean_credit_data(df)
    assert len(result) == 2

def test_clean_credit_data_missing_columns():
    df = pd.DataFrame({"Time": [1], "V1": [0.5]})
    with pytest.raises(ValueError, match="Missing required columns"):
        clean_credit_data(df)


# ─── engineer_features ────────────────────────────────────────

def make_engineered_input():
    """Helper: fraud df ready for feature engineering."""
    df = make_fraud_df()
    df["signup_time"] = pd.to_datetime(df["signup_time"])
    df["purchase_time"] = pd.to_datetime(df["purchase_time"])
    return df

def test_engineer_features_creates_time_since_signup():
    df = make_engineered_input()
    result = engineer_features(df)
    assert "time_since_signup" in result.columns

def test_engineer_features_creates_hour_of_day():
    df = make_engineered_input()
    result = engineer_features(df)
    assert "hour_of_day" in result.columns
    assert result["hour_of_day"].between(0, 23).all()

def test_engineer_features_creates_day_of_week():
    df = make_engineered_input()
    result = engineer_features(df)
    assert "day_of_week" in result.columns
    assert result["day_of_week"].between(0, 6).all()

def test_engineer_features_creates_transaction_count():
    df = make_engineered_input()
    result = engineer_features(df)
    assert "transaction_count" in result.columns
    assert (result["transaction_count"] >= 1).all()

def test_engineer_features_missing_columns():
    df = pd.DataFrame({"user_id": [1], "class": [0]})
    with pytest.raises(ValueError, match="Missing columns"):
        engineer_features(df)


# ─── transform_features ───────────────────────────────────────

def make_transform_input():
    """Helper: df ready for transformation."""
    df = make_engineered_input()
    df = engineer_features(df)
    return df

def test_transform_features_drops_id_columns():
    df = make_transform_input()
    result = transform_features(df, "class")
    for col in ["user_id", "device_id", "ip_address"]:
        assert col not in result.columns

def test_transform_features_encodes_categoricals():
    df = make_transform_input()
    result = transform_features(df, "class")
    for col in ["source", "browser", "sex"]:
        assert col not in result.columns

def test_transform_features_target_col_preserved():
    df = make_transform_input()
    result = transform_features(df, "class")
    assert "class" in result.columns

def test_transform_features_invalid_target():
    df = make_transform_input()
    with pytest.raises(ValueError, match="Target column"):
        transform_features(df, "nonexistent")

def test_transform_features_no_nulls():
    df = make_transform_input()
    result = transform_features(df, "class")
    assert result.isnull().sum().sum() == 0


# ─── handle_imbalance ─────────────────────────────────────────

def test_handle_imbalance_balances_classes():
    X = np.random.rand(100, 5)
    y = np.array([0] * 90 + [1] * 10)
    X_res, y_res = handle_imbalance(X, y)
    counts = pd.Series(y_res).value_counts()
    assert counts[0] == counts[1]

def test_handle_imbalance_returns_correct_shapes():
    X = np.random.rand(100, 5)
    y = np.array([0] * 90 + [1] * 10)
    X_res, y_res = handle_imbalance(X, y)
    assert X_res.shape[1] == 5
    assert len(X_res) == len(y_res)

def test_handle_imbalance_single_class_raises():
    X = np.random.rand(50, 5)
    y = np.zeros(50)
    with pytest.raises(ValueError, match="at least two classes"):
        handle_imbalance(X, y)


# ─── merge_geolocation ────────────────────────────────────────

def make_ip_df():
    """Helper: minimal IP lookup table."""
    return pd.DataFrame({
        "lower_bound_ip_address": [16777216, 3232235520],
        "upper_bound_ip_address": [16777471, 3232235775],
        "country": ["Australia", "Private"],
    })

def test_merge_geolocation_adds_country_column():
    fraud_df = make_fraud_df()
    fraud_df["signup_time"] = pd.to_datetime(fraud_df["signup_time"])
    fraud_df["purchase_time"] = pd.to_datetime(fraud_df["purchase_time"])
    ip_df = make_ip_df()
    result = merge_geolocation(fraud_df, ip_df)
    assert "country" in result.columns

def test_merge_geolocation_unknown_for_unmatched():
    fraud_df = make_fraud_df()
    fraud_df["signup_time"] = pd.to_datetime(fraud_df["signup_time"])
    fraud_df["purchase_time"] = pd.to_datetime(fraud_df["purchase_time"])
    ip_df = make_ip_df()
    result = merge_geolocation(fraud_df, ip_df)
    assert result["country"].notna().all()

def test_merge_geolocation_missing_ip_column():
    fraud_df = make_fraud_df().drop(columns=["ip_address"])
    ip_df = make_ip_df()
    with pytest.raises(ValueError, match="ip_address"):
        merge_geolocation(fraud_df, ip_df)
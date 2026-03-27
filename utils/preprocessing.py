# utils/preprocessing.py
# All data cleaning, encoding, and scaling logic lives here.
# The training notebook calls these functions and then SAVES the fitted scaler.
# The inference API loads the SAME scaler — this guarantees identical transforms.

import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from config import FEATURE_COLS, TARGET_COL, TRAIN_CONFIG, LOCAL_PATHS


# ── Raw data cleaning ────────────────────────────────────────────────────────

def clean_signal_strength(val):
    """Remove ' dBm' suffix and convert to float. E.g. '-55 dBm' → -55.0"""
    if isinstance(val, str):
        return float(val.replace("dBm", "").strip())
    return float(val)


def clean_latency(val):
    """Remove ' ms' suffix. E.g. '34 ms' → 34.0"""
    if isinstance(val, str):
        return float(val.replace("ms", "").strip())
    return float(val)


def clean_bandwidth(val):
    """
    Convert any bandwidth string to Mbps float.
    Handles: '500 Kbps', '10 Mbps', '1.2 Gbps'
    """
    if isinstance(val, str):
        val = val.strip()
        if "Gbps" in val:
            return float(val.replace("Gbps", "").strip()) * 1000
        elif "Kbps" in val:
            return float(val.replace("Kbps", "").strip()) / 1000
        elif "Mbps" in val:
            return float(val.replace("Mbps", "").strip())
    return float(val)


def clean_resource_allocation(val):
    """Convert '75%' → 0.75 (normalized target in [0,1])"""
    if isinstance(val, str):
        return float(val.replace("%", "").strip()) / 100
    return float(val) / 100 if float(val) > 1 else float(val)


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full cleaning pipeline for the raw 5G QoS CSV.
    Returns a cleaned DataFrame with numeric columns ready for scaling.
    """
    df = df.copy()

    # Sort chronologically — critical for LSTM sequence creation
    if "Timestamp" in df.columns:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])
        df = df.sort_values("Timestamp").reset_index(drop=True)

    # Encode categorical application type
    le = LabelEncoder()
    df["Application_Type_encoded"] = le.fit_transform(df["Application_Type"].astype(str))

    # Clean numeric columns
    df["Signal_Strength"] = df["Signal_Strength"].apply(clean_signal_strength)
    df["Latency"] = df["Latency"].apply(clean_latency)
    df["Required_Bandwidth"] = df["Required_Bandwidth"].apply(clean_bandwidth)
    df["Allocated_Bandwidth"] = df["Allocated_Bandwidth"].apply(clean_bandwidth)
    df[TARGET_COL] = df[TARGET_COL].apply(clean_resource_allocation)

    # Drop rows with any NaN in the columns we use
    cols_needed = FEATURE_COLS + [TARGET_COL]
    df = df.dropna(subset=cols_needed).reset_index(drop=True)

    return df


# ── Scaling ──────────────────────────────────────────────────────────────────

def fit_and_save_scaler(df: pd.DataFrame, save_path: str = LOCAL_PATHS["scaler"]) -> MinMaxScaler:
    """
    Fit MinMaxScaler on the feature columns of df and save it to disk.
    Call this ONCE during training. The saved scaler is then loaded at inference.
    """
    scaler = MinMaxScaler()
    scaler.fit(df[FEATURE_COLS])
    joblib.dump(scaler, save_path)
    print(f"Scaler saved to {save_path}")
    return scaler


def load_scaler(load_path: str = LOCAL_PATHS["scaler"]) -> MinMaxScaler:
    """Load a previously fitted scaler from disk."""
    return joblib.load(load_path)


def scale_features(df: pd.DataFrame, scaler: MinMaxScaler) -> np.ndarray:
    """Apply a fitted scaler to the feature columns of df. Returns numpy array."""
    return scaler.transform(df[FEATURE_COLS])


# ── Sequence generation ──────────────────────────────────────────────────────

def create_sequences(features: np.ndarray, targets: np.ndarray, seq_len: int = TRAIN_CONFIG["sequence_length"]):
    """
    Sliding window sequence generator for LSTM input.

    For each index i in [seq_len, len(features)]:
        X[i] = features[i-seq_len : i]   shape: (seq_len, n_features)
        y[i] = targets[i]                 scalar

    Returns:
        X: np.ndarray shape (n_samples, seq_len, n_features)
        y: np.ndarray shape (n_samples,)
    """
    X, y = [], []
    for i in range(seq_len, len(features)):
        X.append(features[i - seq_len: i])
        y.append(targets[i])
    return np.array(X), np.array(y)


# ── Preprocess a single inference input ──────────────────────────────────────

def preprocess_single_input(raw_input: dict, scaler: MinMaxScaler, le_classes: list) -> np.ndarray:
    """
    Convert a raw API input dict to a scaled feature vector (1 time step).
    The API must call this for each time step and stack seq_len of them into a sequence.

    Args:
        raw_input: dict with keys matching raw column names (e.g. 'Signal_Strength')
        scaler: fitted MinMaxScaler loaded from disk
        le_classes: list of LabelEncoder classes (application types), in order

    Returns:
        numpy array of shape (1, n_features) — one scaled feature row
    """
    app_type = raw_input.get("Application_Type", "")
    if app_type in le_classes:
        app_encoded = le_classes.index(app_type)
    else:
        app_encoded = 0  # fallback for unknown types

    row = {
        "Signal_Strength": clean_signal_strength(raw_input["Signal_Strength"]),
        "Latency": clean_latency(raw_input["Latency"]),
        "Required_Bandwidth": clean_bandwidth(raw_input["Required_Bandwidth"]),
        "Allocated_Bandwidth": clean_bandwidth(raw_input["Allocated_Bandwidth"]),
        "Application_Type_encoded": app_encoded,
    }

    df_row = pd.DataFrame([row])[FEATURE_COLS]
    return scaler.transform(df_row)

# config.py
# Central configuration for all hyperparameters and settings.
# Both training and inference import from here, so they always stay in sync.

# ── Model Architecture ──────────────────────────────────────────────────────
MODEL_CONFIG = {
    "input_size": 5,        # Number of input features after preprocessing
    "hidden_size": 64,      # Hidden units per LSTM layer
    "num_layers": 2,        # Number of stacked LSTM layers (MUST match training)
    "output_size": 1,       # Single regression output: resource allocation %
    "dropout": 0.2,         # Dropout between LSTM layers (only active during training)
}

# ── Training ────────────────────────────────────────────────────────────────
TRAIN_CONFIG = {
    "sequence_length": 10,  # Number of time steps per input window
    "batch_size": 32,
    "epochs": 100,
    "learning_rate": 1e-3,
    "early_stopping_patience": 10,
    "train_val_split": 0.8,  # 80% train, 20% val (from the 80% training portion)
    "test_split": 0.2,       # 20% held out as final test set
}

# ── AWS S3 ──────────────────────────────────────────────────────────────────
AWS_CONFIG = {
    "region_name": "eu-north-1",
    "bucket_name": "my-lstm-models",
    "model_key": "best_model.pth",
    "scaler_key": "scaler.pkl",        # NEW: scaler is also saved/loaded from S3
}

# ── Local Paths ─────────────────────────────────────────────────────────────
LOCAL_PATHS = {
    "model": "models/best_model.pth",
    "scaler": "models/scaler.pkl",
    "dataset": "data/5g_qos_dataset.csv"
,
}

# ── Feature Columns ─────────────────────────────────────────────────────────
# These must match exactly what was used during training preprocessing
FEATURE_COLS = [
    "Signal_Strength",
    "Latency",
    "Required_Bandwidth",
    "Allocated_Bandwidth",
    "Application_Type_encoded",
]
TARGET_COL = "Resource_Allocation"

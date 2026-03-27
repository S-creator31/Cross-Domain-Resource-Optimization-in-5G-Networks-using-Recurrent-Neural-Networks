# test_on_realdata.py
# Trains LSTM on Kaggle dataset, tests on your real collected data.
# No user input needed — reads directly from real_network_data.csv

import sys
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ".")

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from config import MODEL_CONFIG, TRAIN_CONFIG, LOCAL_PATHS, FEATURE_COLS, TARGET_COL
from utils.model import LSTMModel
from utils.preprocessing import (
    preprocess_dataframe, load_scaler, scale_features, create_sequences
)

# ── Configuration ─────────────────────────────────────────────────────────────
KAGGLE_CSV   = "data/5g_qos_dataset.csv"
REALDATA_CSV = "data/real_network_data.csv"
SEQ_LEN      = TRAIN_CONFIG["sequence_length"]

# ═══════════════════════════════════════════════════════════════════════════════
# 1. Load trained model and scaler (from Kaggle training)
# ═══════════════════════════════════════════════════════════════════════════════
print("="*55)
print("STEP 1: Loading trained model and scaler")
print("="*55)

model = LSTMModel(**MODEL_CONFIG)
try:
    model.load_state_dict(
        torch.load(LOCAL_PATHS["model"], map_location="cpu"),
        strict=True
    )
    model.eval()
    print(f"Model loaded from {LOCAL_PATHS['model']}")
except Exception as e:
    print(f"ERROR loading model: {e}")
    print("Run notebooks/LSTM_Train.py first to train on Kaggle data.")
    exit(1)

try:
    scaler = load_scaler(LOCAL_PATHS["scaler"])
    print(f"Scaler loaded from {LOCAL_PATHS['scaler']}")
except Exception as e:
    print(f"ERROR loading scaler: {e}")
    exit(1)

# ═══════════════════════════════════════════════════════════════════════════════
# 2. Load and preprocess real collected data
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*55)
print("STEP 2: Loading real network data")
print("="*55)

try:
    df_real = pd.read_csv(REALDATA_CSV)
    print(f"Real data loaded: {df_real.shape[0]} rows")
    print(f"Columns: {list(df_real.columns)}")
except FileNotFoundError:
    print(f"ERROR: {REALDATA_CSV} not found.")
    print("Run data_collector.py first to collect real data.")
    exit(1)

# Preprocess using same pipeline as training
df_real_clean = preprocess_dataframe(df_real)
print(f"After cleaning: {df_real_clean.shape[0]} rows")

if df_real_clean.shape[0] < SEQ_LEN + 1:
    print(f"ERROR: Need at least {SEQ_LEN + 1} rows. You have {df_real_clean.shape[0]}.")
    print("Run data_collector.py longer to collect more data.")
    exit(1)

# ── Scale using the SAME scaler fitted on Kaggle data ─────────────────────────
# This is critical — we must use the Kaggle scaler, not refit on real data
# because the model was trained on Kaggle-scaled features
features_real = scaler.transform(df_real_clean[FEATURE_COLS])
targets_real  = df_real_clean[TARGET_COL].values
print(f"Features scaled using Kaggle scaler (same as training)")

# ═══════════════════════════════════════════════════════════════════════════════
# 3. Create sequences from real data
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*55)
print("STEP 3: Creating sequences")
print("="*55)

X_real, y_real = create_sequences(features_real, targets_real, seq_len=SEQ_LEN)
print(f"Sequences created: {len(X_real)}")
print(f"Each sequence shape: {X_real[0].shape}")

# Convert to tensors — shape: (seq_len, batch, features)
X_real_t = torch.FloatTensor(X_real).permute(1, 0, 2)
y_real_t  = torch.FloatTensor(y_real).unsqueeze(1)

# ═══════════════════════════════════════════════════════════════════════════════
# 4. Run inference on real data
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*55)
print("STEP 4: Running inference on real data")
print("="*55)

model.eval()
all_preds = []

# Run one sequence at a time so we get a prediction per time step
with torch.no_grad():
    for i in range(X_real_t.shape[1]):
        single = X_real_t[:, i:i+1, :]  # (seq_len, 1, features)
        pred   = model(single).item()
        all_preds.append(pred)

preds    = np.array(all_preds)
y_true   = y_real[:len(preds)]

print(f"Total predictions made: {len(preds)}")

# ═══════════════════════════════════════════════════════════════════════════════
# 5. Evaluation metrics
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*55)
print("STEP 5: Evaluation on real data")
print("="*55)

mae  = mean_absolute_error(y_true, preds)
mse  = mean_squared_error(y_true, preds)
r2   = r2_score(y_true, preds)

print(f"\n  MAE  : {mae:.4f}  ({mae*100:.1f}% average error)")
print(f"  MSE  : {mse:.4f}")
print(f"  R²   : {r2:.4f}")

# Compare to Kaggle test performance
print("\n  Context (Kaggle test results from training):")
print("  MAE  : 0.1500")
print("  MSE  : 0.0372")
print("  R²   : 0.2357")

if r2 > 0.2357:
    print("\n  Result: Model performs BETTER on real data than Kaggle test set.")
elif r2 > 0:
    print("\n  Result: Model generalizes reasonably to real data.")
else:
    print("\n  Result: Model struggles on real data — likely due to")
    print("          distribution mismatch between Kaggle and real data.")
    print("          Try combining both datasets for retraining.")

# ═══════════════════════════════════════════════════════════════════════════════
# 6. Per application type breakdown on real data
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*55)
print("STEP 6: Per-application-type breakdown")
print("="*55)

app_types = df_real_clean["Application_Type"].values[SEQ_LEN:SEQ_LEN+len(preds)]
results_df = pd.DataFrame({
    "Application_Type": app_types,
    "true":  y_true,
    "pred":  preds,
    "error": np.abs(y_true - preds)
})

summary = results_df.groupby("Application_Type").agg(
    Count=("error", "count"),
    MAE=("error", "mean"),
).round(4).sort_values("MAE", ascending=False)
print(summary.to_string())

# ═══════════════════════════════════════════════════════════════════════════════
# 7. Plots
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*55)
print("STEP 7: Generating plots")
print("="*55)

os.makedirs("outputs", exist_ok=True)
n = min(100, len(preds))

fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# Plot 1: Predicted vs actual
axes[0].plot(y_true[:n],  label="True (real data)",      linewidth=1.2)
axes[0].plot(preds[:n],   label="Predicted (LSTM)",      linewidth=1.2, linestyle="--")
axes[0].set_title(f"Real Data Test — Predicted vs Actual Resource Allocation\nMAE={mae:.4f}  R²={r2:.4f}")
axes[0].set_xlabel("Sample")
axes[0].set_ylabel("Resource Allocation")
axes[0].legend()

# Plot 2: Error over time
errors = np.abs(y_true[:n] - preds[:n])
axes[1].fill_between(range(n), errors, alpha=0.6, color="coral")
axes[1].axhline(y=mae, color="red", linestyle="--", label=f"Mean Error = {mae:.4f}")
axes[1].set_title("Prediction Error Over Time")
axes[1].set_xlabel("Sample")
axes[1].set_ylabel("Absolute Error")
axes[1].legend()

plt.tight_layout()
plt.savefig("outputs/realdata_test_results.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: outputs/realdata_test_results.png")

# Per-app bar chart
if len(summary) > 1:
    fig2, ax = plt.subplots(figsize=(8, 4))
    summary["MAE"].plot(kind="bar", ax=ax, color="steelblue", edgecolor="white")
    ax.set_title("MAE by Application Type — Real Data Test")
    ax.set_ylabel("MAE")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig("outputs/realdata_per_app_error.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: outputs/realdata_per_app_error.png")

print("\nDone. All results saved to outputs/")
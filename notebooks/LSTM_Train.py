# notebooks/LSTM_Train.py
# Full training pipeline. Run this to produce best_model.pth and scaler.pkl.
# Written as a .py script so it runs cleanly; convert to .ipynb with jupytext if needed.
#
# Key changes vs original notebook:
#   1. Uses LSTMModel from utils/model.py (2-layer, with dropout) instead of redefining it
#   2. Saves the fitted MinMaxScaler alongside the model
#   3. Uploads BOTH artifacts to S3 after training
#   4. Adds baseline (linear regression) comparison
#   5. Adds per-application-type error breakdown
#   6. Adds model variant comparison (1-layer vs 2-layer)

import sys
sys.path.insert(0, "..")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from config import MODEL_CONFIG, TRAIN_CONFIG, LOCAL_PATHS, FEATURE_COLS, TARGET_COL
from utils.model import LSTMModel
from utils.preprocessing import (
    preprocess_dataframe, fit_and_save_scaler, scale_features, create_sequences
)
from utils.aws_utils import upload_model_and_scaler

import os
os.makedirs("../models", exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════════
# 1. Load and preprocess data
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("STEP 1: Data Preprocessing")
print("=" * 60)

df_raw = pd.read_csv("data/real_network_data.csv") 
print(f"Raw dataset shape: {df_raw.shape}")
print(f"Columns: {list(df_raw.columns)}\n")

df = preprocess_dataframe(df_raw)
print(f"Cleaned dataset shape: {df.shape}")

# Fit and save the scaler — this is the critical new step
# The same scaler object is loaded at inference time
scaler = fit_and_save_scaler(df, save_path=LOCAL_PATHS["scaler"])

features_scaled = scale_features(df, scaler)
targets = df[TARGET_COL].values

# ═══════════════════════════════════════════════════════════════════════════════
# 2. EDA — distributions and correlation
#    (See notebooks/EDA.py for the full EDA; here we just print a summary)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 2: Quick data summary")
print("=" * 60)
print(df[FEATURE_COLS + [TARGET_COL]].describe().round(3))

# ═══════════════════════════════════════════════════════════════════════════════
# 3. Sequence generation and train/val/test split
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 3: Sequence generation")
print("=" * 60)

SEQ_LEN = TRAIN_CONFIG["sequence_length"]
X, y = create_sequences(features_scaled, targets, seq_len=SEQ_LEN)
print(f"Total sequences: {len(X)}  |  X shape: {X.shape}  |  y shape: {y.shape}")

# First 80% → train+val, last 20% → test (no shuffle — preserves time order)
test_split_idx = int(len(X) * (1 - TRAIN_CONFIG["test_split"]))
X_trainval, X_test = X[:test_split_idx], X[test_split_idx:]
y_trainval, y_test = y[:test_split_idx], y[test_split_idx:]

val_split_idx = int(len(X_trainval) * TRAIN_CONFIG["train_val_split"])
X_train, X_val = X_trainval[:val_split_idx], X_trainval[val_split_idx:]
y_train, y_val = y_trainval[:val_split_idx], y_trainval[val_split_idx:]

print(f"Train: {len(X_train)}  |  Val: {len(X_val)}  |  Test: {len(X_test)}")

# Convert to tensors — shape: (seq_len, batch, features) for LSTM
def to_tensor(X, y):
    xt = torch.FloatTensor(X).permute(1, 0, 2)  # (batch, seq, feat) → (seq, batch, feat)
    yt = torch.FloatTensor(y).unsqueeze(1)
    return xt, yt

X_train_t, y_train_t = to_tensor(X_train, y_train)
X_val_t, y_val_t     = to_tensor(X_val, y_val)
X_test_t, y_test_t   = to_tensor(X_test, y_test)

train_loader = DataLoader(
    TensorDataset(X_train_t.permute(1,0,2), y_train_t),
    batch_size=TRAIN_CONFIG["batch_size"],
    shuffle=True
)

# ═══════════════════════════════════════════════════════════════════════════════
# 4. Training function (reused for both 1-layer and 2-layer models)
# ═══════════════════════════════════════════════════════════════════════════════

def train_model(model, label="Model"):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=TRAIN_CONFIG["learning_rate"])
    best_val_loss = float("inf")
    patience_counter = 0
    train_losses, val_losses = [], []

    for epoch in range(TRAIN_CONFIG["epochs"]):
        model.train()
        epoch_loss = 0
        for xb, yb in train_loader:
            xb = xb.permute(1, 0, 2)  # back to (seq, batch, feat)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = criterion(val_pred, y_val_t).item()

        train_losses.append(epoch_loss / len(train_loader))
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), LOCAL_PATHS["model"])
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0:
            print(f"  [{label}] Epoch {epoch+1:3d} | Train Loss: {train_losses[-1]:.4f} | Val Loss: {val_loss:.4f}")

        if patience_counter >= TRAIN_CONFIG["early_stopping_patience"]:
            print(f"  [{label}] Early stopping at epoch {epoch+1}")
            break

    return train_losses, val_losses


def evaluate_model(model, X_t, y_true_t):
    model.eval()
    with torch.no_grad():
        preds = model(X_t).numpy().flatten()
    y_true = y_true_t.numpy().flatten()
    return {
        "MAE":  mean_absolute_error(y_true, preds),
        "MSE":  mean_squared_error(y_true, preds),
        "R2":   r2_score(y_true, preds),
        "preds": preds,
        "true":  y_true,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 5. ADDITION: Model variant comparison — 1-layer vs 2-layer LSTM
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 4: Model variant comparison (1-layer vs 2-layer)")
print("=" * 60)

model_1layer = LSTMModel(**{**MODEL_CONFIG, "num_layers": 1, "dropout": 0.0})
model_2layer = LSTMModel(**MODEL_CONFIG)  # num_layers=2 by default from config

print("\nTraining 1-layer LSTM...")
train_model(model_1layer, label="1-layer")
res_1layer = evaluate_model(model_1layer, X_test_t, y_test_t)

# Reload best weights for 2-layer (saved by train_model)
print("\nTraining 2-layer LSTM...")
train_model(model_2layer, label="2-layer")
model_2layer.load_state_dict(torch.load(LOCAL_PATHS["model"]))
res_2layer = evaluate_model(model_2layer, X_test_t, y_test_t)

# ── Baseline: Linear Regression ──────────────────────────────────────────────
lr = LinearRegression()
lr.fit(X_train.reshape(len(X_train), -1), y_train)
lr_preds = lr.predict(X_test.reshape(len(X_test), -1))
res_lr = {
    "MAE": mean_absolute_error(y_test, lr_preds),
    "MSE": mean_squared_error(y_test, lr_preds),
    "R2":  r2_score(y_test, lr_preds),
}

print("\n")
print("┌──────────────────────┬─────────┬─────────┬─────────┐")
print("│ Model                │   MAE   │   MSE   │   R²    │")
print("├──────────────────────┼─────────┼─────────┼─────────┤")
print(f"│ Linear Regression    │ {res_lr['MAE']:.4f}  │ {res_lr['MSE']:.4f}  │ {res_lr['R2']:.4f}  │")
print(f"│ LSTM (1 layer)       │ {res_1layer['MAE']:.4f}  │ {res_1layer['MSE']:.4f}  │ {res_1layer['R2']:.4f}  │")
print(f"│ LSTM (2 layers) ✓    │ {res_2layer['MAE']:.4f}  │ {res_2layer['MSE']:.4f}  │ {res_2layer['R2']:.4f}  │")
print("└──────────────────────┴─────────┴─────────┴─────────┘")

# ── Plot: Predicted vs actual ─────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, res, title in zip(axes, [res_1layer, res_2layer], ["1-layer LSTM", "2-layer LSTM"]):
    n = min(80, len(res["true"]))
    ax.plot(res["true"][:n], label="True", linewidth=1.2)
    ax.plot(res["preds"][:n], label="Predicted", linewidth=1.2, linestyle="--")
    ax.set_title(f"{title}  |  R²={res['R2']:.4f}")
    ax.set_xlabel("Sample")
    ax.set_ylabel("Resource Allocation")
    ax.legend()
plt.suptitle("True vs Predicted Resource Allocation", fontsize=13)
plt.tight_layout()
plt.savefig("outputs/model_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: outputs/model_comparison.png")

# ═══════════════════════════════════════════════════════════════════════════════
# 6. ADDITION: Per-application-type error breakdown
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 5: Per-application-type error breakdown")
print("=" * 60)

# Align test predictions with original df rows (offset by sequence length)
app_types_test = df["Application_Type"].values[
    test_split_idx + SEQ_LEN: test_split_idx + SEQ_LEN + len(res_2layer["preds"])
]
# If alignment goes out of bounds, trim
n = min(len(app_types_test), len(res_2layer["preds"]))
app_types_test = app_types_test[:n]
preds_test = res_2layer["preds"][:n]
true_test  = res_2layer["true"][:n]

per_app_df = pd.DataFrame({
    "Application_Type": app_types_test,
    "true": true_test,
    "pred": preds_test,
})
per_app_df["abs_error"] = (per_app_df["true"] - per_app_df["pred"]).abs()
per_app_df["squared_error"] = per_app_df["abs_error"] ** 2

summary = per_app_df.groupby("Application_Type").agg(
    Count=("abs_error", "count"),
    MAE=("abs_error", "mean"),
    MSE=("squared_error", "mean"),
).round(4).sort_values("MAE", ascending=False)
print(summary.to_string())

fig, ax = plt.subplots(figsize=(9, 5))
summary["MAE"].plot(kind="bar", ax=ax, color="steelblue", edgecolor="white")
ax.set_title("Mean Absolute Error by Application Type (2-layer LSTM)", fontsize=12)
ax.set_ylabel("MAE")
ax.set_xlabel("Application Type")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig("outputs/per_app_error.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: outputs/per_app_error.png")

# ═══════════════════════════════════════════════════════════════════════════════
# 7. Upload to S3
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 6: Uploading model and scaler to S3")
print("=" * 60)
# Uncomment when AWS credentials are configured:
# upload_model_and_scaler()
print("(Upload skipped — uncomment upload_model_and_scaler() when ready)")

# notebooks/EDA.py
# Exploratory Data Analysis for the 5G QoS dataset.
# Run this before training to understand the data distributions,
# class balance, correlations, and feature relationships.
#
# This is an ADDITION — the original project had no EDA.

import sys
sys.path.insert(0, "..")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os

from utils.preprocessing import preprocess_dataframe
from config import FEATURE_COLS, TARGET_COL

os.makedirs("../outputs", exist_ok=True)

# ── Load data ────────────────────────────────────────────────────────────────
df_raw = pd.read_csv("../data/5g_qos_dataset.csv")
df = preprocess_dataframe(df_raw)
print(f"Dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(df[FEATURE_COLS + [TARGET_COL]].describe().round(3))

# ═══════════════════════════════════════════════════════════════════════════════
# Plot 1: Feature distributions
# ═══════════════════════════════════════════════════════════════════════════════
numeric_cols = ["Signal_Strength", "Latency", "Required_Bandwidth", "Allocated_Bandwidth", TARGET_COL]

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()

for i, col in enumerate(numeric_cols):
    axes[i].hist(df[col].dropna(), bins=40, color="steelblue", edgecolor="white", alpha=0.85)
    axes[i].set_title(col, fontsize=11)
    axes[i].set_xlabel("Value")
    axes[i].set_ylabel("Frequency")

# Hide the 6th unused subplot
axes[5].set_visible(False)

plt.suptitle("Feature Distributions", fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig("../outputs/eda_distributions.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: outputs/eda_distributions.png")

# ═══════════════════════════════════════════════════════════════════════════════
# Plot 2: QoS class balance
# ═══════════════════════════════════════════════════════════════════════════════
if "QoS" in df_raw.columns:
    qos_counts = df_raw["QoS"].value_counts()
    colors = ["#2ecc71", "#3498db", "#f39c12", "#e74c3c"]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(qos_counts.index, qos_counts.values, color=colors[:len(qos_counts)], edgecolor="white")
    for bar, val in zip(bars, qos_counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 100,
                f"{val:,}", ha="center", va="bottom", fontsize=10)
    ax.set_title("QoS Class Distribution", fontsize=13)
    ax.set_xlabel("QoS Class")
    ax.set_ylabel("Count")
    plt.tight_layout()
    plt.savefig("../outputs/eda_qos_balance.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: outputs/eda_qos_balance.png")

# ═══════════════════════════════════════════════════════════════════════════════
# Plot 3: Correlation heatmap
# ═══════════════════════════════════════════════════════════════════════════════
corr = df[FEATURE_COLS + [TARGET_COL]].corr().round(2)

fig, ax = plt.subplots(figsize=(8, 6))
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)  # upper triangle only
sns.heatmap(
    corr, annot=True, fmt=".2f", cmap="coolwarm",
    center=0, square=True, linewidths=0.5,
    ax=ax, cbar_kws={"shrink": 0.8}
)
ax.set_title("Feature Correlation Matrix", fontsize=13)
plt.tight_layout()
plt.savefig("../outputs/eda_correlation.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: outputs/eda_correlation.png")

# ═══════════════════════════════════════════════════════════════════════════════
# Plot 4: Resource allocation by application type (box plot)
# ═══════════════════════════════════════════════════════════════════════════════
if "Application_Type" in df.columns:
    fig, ax = plt.subplots(figsize=(10, 5))
    app_order = df.groupby("Application_Type")[TARGET_COL].median().sort_values(ascending=False).index
    df.boxplot(column=TARGET_COL, by="Application_Type", ax=ax,
               order=app_order, patch_artist=True,
               boxprops=dict(facecolor="steelblue", alpha=0.7))
    ax.set_title("Resource Allocation by Application Type", fontsize=12)
    ax.set_xlabel("Application Type")
    ax.set_ylabel("Resource Allocation (normalized)")
    plt.xticks(rotation=25, ha="right")
    plt.suptitle("")  # remove auto-suptitle from boxplot
    plt.tight_layout()
    plt.savefig("../outputs/eda_by_app_type.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: outputs/eda_by_app_type.png")

# ═══════════════════════════════════════════════════════════════════════════════
# Plot 5: Resource allocation over time (first 500 rows)
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(df[TARGET_COL].values[:500], linewidth=0.8, color="steelblue")
ax.set_title("Resource Allocation Over Time (first 500 samples)", fontsize=12)
ax.set_xlabel("Sample index")
ax.set_ylabel("Resource Allocation (normalized)")
plt.tight_layout()
plt.savefig("../outputs/eda_timeseries.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: outputs/eda_timeseries.png")

print("\nEDA complete. All plots saved to outputs/")

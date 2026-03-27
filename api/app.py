# api/app.py
# Flask REST API for 5G resource allocation prediction.
# LOCAL VERSION — no AWS required. Loads model and scaler directly from models/ folder.

import sys
import os

# Automatically set working directory to 5g_project/ root
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ".")

import torch
import numpy as np
from flask import Flask, request, jsonify

from config import MODEL_CONFIG, LOCAL_PATHS, TRAIN_CONFIG
from utils.model import LSTMModel
from utils.preprocessing import load_scaler, preprocess_single_input

app = Flask(__name__)

# ── Global model and scaler (loaded once at startup) ─────────────────────────
model = None
scaler = None

# Must match the unique values of Application_Type in your dataset
# Run: print(df["Application_Type"].unique()) after loading data to verify
APP_TYPE_CLASSES = ["Gaming", "Streaming", "VoIP", "Browsing", "File Transfer"]


def load_model_and_scaler():
    """
    Load model and scaler directly from local models/ folder.
    No AWS required — just run training first to generate these files.
    """
    global model, scaler

    try:
        m = LSTMModel(
            input_size=MODEL_CONFIG["input_size"],
            hidden_size=MODEL_CONFIG["hidden_size"],
            num_layers=MODEL_CONFIG["num_layers"],
            output_size=MODEL_CONFIG["output_size"],
            dropout=MODEL_CONFIG["dropout"],
        )
        m.load_state_dict(
            torch.load(LOCAL_PATHS["model"], map_location=torch.device("cpu")),
            strict=True,
        )
        m.eval()
        model = m
        print(f"Model loaded from {LOCAL_PATHS['model']}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

    try:
        scaler = load_scaler(LOCAL_PATHS["scaler"])
        print(f"Scaler loaded from {LOCAL_PATHS['scaler']}")
    except Exception as e:
        print(f"Error loading scaler: {e}")
        return False

    return True


# ── Health check ──────────────────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    """Check if the model and scaler are loaded and ready."""
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
    })


# ── Prediction endpoint ───────────────────────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    """
    Accepts a JSON body with a list of 10 consecutive time steps.

    Example request body:
    {
        "sequence": [
            {
                "Signal_Strength": "-65 dBm",
                "Latency": "30 ms",
                "Required_Bandwidth": "20 Mbps",
                "Allocated_Bandwidth": "18 Mbps",
                "Application_Type": "Streaming"
            },
            ... (9 more steps, 10 total)
        ]
    }

    Returns:
    {
        "predicted_resource_allocation": 0.72,
        "predicted_resource_allocation_percent": "72.0%"
    }
    """
    if model is None or scaler is None:
        return jsonify({"error": "Model not loaded. Check terminal for errors."}), 503

    try:
        data = request.get_json()
        sequence = data.get("sequence", [])
        seq_len = TRAIN_CONFIG["sequence_length"]

        if len(sequence) != seq_len:
            return jsonify({
                "error": f"Expected {seq_len} time steps, got {len(sequence)}"
            }), 400

        # Preprocess each time step using the same scaler as training
        steps = []
        for step in sequence:
            scaled_row = preprocess_single_input(step, scaler, APP_TYPE_CLASSES)
            steps.append(scaled_row[0])

        # Build tensor of shape (seq_len, 1, n_features)
        input_tensor = torch.FloatTensor(np.array(steps)).unsqueeze(1)

        with torch.no_grad():
            prediction = model(input_tensor).item()

        return jsonify({
            "predicted_resource_allocation": round(prediction, 4),
            "predicted_resource_allocation_percent": f"{prediction * 100:.1f}%",
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Demo inference ────────────────────────────────────────────────────────────
def run_demo_inference():
    """Quick sanity check with random input after loading."""
    if model is None:
        print("Model not loaded.")
        return

    seq_len = TRAIN_CONFIG["sequence_length"]
    n_features = MODEL_CONFIG["input_size"]
    sample_input = torch.randn(seq_len, 1, n_features)

    with torch.no_grad():
        output = model(sample_input)

    print(f"Demo prediction: {output.item():.4f}  ({output.item()*100:.1f}%)")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if load_model_and_scaler():
        print("\nRunning demo inference...")
        run_demo_inference()
        print("\nStarting Flask API on http://0.0.0.0:5000")
        app.run(host="0.0.0.0", port=5000, debug=False)
    else:
        print("\nFailed to load model or scaler.")
        print("Make sure you have run notebooks/LSTM_Train.py first.")
        print("Expected files:")
        print(f"  - {LOCAL_PATHS['model']}")
        print(f"  - {LOCAL_PATHS['scaler']}")

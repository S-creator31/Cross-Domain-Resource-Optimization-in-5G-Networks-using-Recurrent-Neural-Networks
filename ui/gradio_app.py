# ui/gradio_app.py
# Simple Gradio interface for the 5G resource allocation predictor.
# This is an ADDITION — the original project had no demo UI.
#
# Run this AFTER the Flask API is running (api/app.py).
# Or run it standalone in LOCAL_MODE to use the model directly without the API.

import sys
import os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ".")

import gradio as gr
import torch
import numpy as np
import requests
import json

from config import MODEL_CONFIG, TRAIN_CONFIG, LOCAL_PATHS
from utils.model import LSTMModel
from utils.preprocessing import load_scaler, preprocess_single_input

# ── Configuration ─────────────────────────────────────────────────────────────
API_URL = "http://localhost:5000/predict"
LOCAL_MODE = True   # Set False to call the Flask API instead

APP_TYPE_CLASSES = ["Gaming", "Streaming", "VoIP", "Browsing", "File Transfer"]

# ── Load model locally (used only in LOCAL_MODE) ──────────────────────────────
_local_model = None
_local_scaler = None

def load_local():
    global _local_model, _local_scaler
    if _local_model is not None:
        return True
    try:
        _local_scaler = load_scaler(LOCAL_PATHS["scaler"])
        m = LSTMModel(**MODEL_CONFIG)
        m.load_state_dict(torch.load(LOCAL_PATHS["model"], map_location="cpu"), strict=True)
        m.eval()
        _local_model = m
        return True
    except Exception as e:
        print(f"Could not load local model: {e}")
        return False


def predict_local(rows):
    """Run inference directly using the local model (no API call needed)."""
    if not load_local():
        return "Error: model/scaler not loaded. Run training first."

    steps = []
    for r in rows:
        scaled = preprocess_single_input(r, _local_scaler, APP_TYPE_CLASSES)
        steps.append(scaled[0])

    arr = torch.FloatTensor(np.array(steps)).unsqueeze(1)  # (seq_len, 1, n_feat)
    with torch.no_grad():
        pred = _local_model(arr).item()
    return pred


def predict_via_api(rows):
    """Call the Flask REST API for prediction."""
    resp = requests.post(API_URL, json={"sequence": rows}, timeout=10)
    resp.raise_for_status()
    return resp.json()["predicted_resource_allocation"]


# ── Core prediction function (called by Gradio) ───────────────────────────────
def predict(signal_strength, latency, req_bw, alloc_bw, app_type):
    """
    The Gradio UI collects one set of values and repeats it seq_len times.
    In a real deployment you'd collect 10 consecutive real readings.
    """
    row = {
        "Signal_Strength": f"{signal_strength} dBm",
        "Latency": f"{latency} ms",
        "Required_Bandwidth": f"{req_bw} Mbps",
        "Allocated_Bandwidth": f"{alloc_bw} Mbps",
        "Application_Type": app_type,
    }
    seq_len = TRAIN_CONFIG["sequence_length"]
    rows = [row] * seq_len  # repeat to fill the 10-step window

    try:
        if LOCAL_MODE:
            pred = predict_local(rows)
        else:
            pred = predict_via_api(rows)
    except Exception as e:
        return f"Error: {e}", ""

    bar = "█" * int(pred * 20) + "░" * (20 - int(pred * 20))
    label = (
        "🟢 Excellent" if pred < 0.4 else
        "🟡 Good"      if pred < 0.6 else
        "🟠 Fair"      if pred < 0.8 else
        "🔴 High Load"
    )
    summary = f"{label}  |  Predicted allocation: {pred*100:.1f}%\n[{bar}]"
    return f"{pred*100:.1f}%", summary


# ── Gradio layout ─────────────────────────────────────────────────────────────
with gr.Blocks(title="5G Resource Allocation Predictor") as demo:
    gr.Markdown("## 5G Resource Allocation Predictor\nEnter current network metrics to predict resource allocation.")

    with gr.Row():
        with gr.Column():
            signal_strength = gr.Slider(-120, -30, value=-65, step=1, label="Signal Strength (dBm)")
            latency         = gr.Slider(1, 200, value=30, step=1, label="Latency (ms)")
            req_bw          = gr.Slider(0.1, 100, value=20, step=0.1, label="Required Bandwidth (Mbps)")
            alloc_bw        = gr.Slider(0.1, 100, value=18, step=0.1, label="Allocated Bandwidth (Mbps)")
            app_type        = gr.Dropdown(APP_TYPE_CLASSES, value="Streaming", label="Application Type")
            submit_btn      = gr.Button("Predict", variant="primary")

        with gr.Column():
            pct_out     = gr.Textbox(label="Predicted Resource Allocation")
            summary_out = gr.Textbox(label="Status", lines=3)

    submit_btn.click(
        fn=predict,
        inputs=[signal_strength, latency, req_bw, alloc_bw, app_type],
        outputs=[pct_out, summary_out],
    )

    gr.Markdown(
        "**Note:** In LOCAL_MODE the UI uses the model directly. "
        "Set `LOCAL_MODE = False` to route through the Flask API instead."
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)

# 5G Traffic Prediction Using Machine Learning and Cloud Integration

## Authors
- Amisha Lalwani (202251013) · Aradhya Verma (202251022)
- Dhwani Saliya (202251041) · Gaurav Barhate (202251049)

---

## Project Structure

```
5g_project/
├── config.py                   # Central hyperparameters — edit once, applies everywhere
├── requirements.txt            # All dependencies with pinned versions
│
├── data/
│   └── 5g_qos_dataset.csv      # Kaggle 5G QoS dataset (download separately)
│
├── models/                     # Created automatically during training
│   ├── best_model.pth          # Saved LSTM weights
│   └── scaler.pkl              # Saved MinMaxScaler (NEW)
│
├── notebooks/
│   ├── EDA.py                  # Exploratory data analysis (NEW)
│   └── LSTM_Train.py           # Full training pipeline
│
├── utils/
│   ├── model.py                # LSTMModel class — single source of truth (FIXED)
│   ├── preprocessing.py        # Cleaning, scaling, sequence generation (NEW)
│   └── aws_utils.py            # S3 upload/download for model + scaler (UPDATED)
│
├── api/
│   └── app.py                  # Flask REST API with /predict endpoint (FIXED + NEW)
│
├── ui/
│   └── gradio_app.py           # Gradio demo UI (NEW)
│
└── outputs/                    # Plots saved here by EDA and training scripts
```

---

## Setup

```bash
pip install -r requirements.txt
```

Download the [5G QoS dataset](https://www.kaggle.com/datasets/omarsobhy14/5g-quality-of-service) and place it at `data/5g_qos_dataset.csv`.

Configure AWS credentials:
```bash
aws configure   # or set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY env vars
```

---

## Running the Project

### 1. Exploratory Data Analysis
```bash
cd notebooks
python EDA.py
# Outputs saved to outputs/
```

### 2. Train the model
```bash
cd notebooks
python LSTM_Train.py
# Saves models/best_model.pth and models/scaler.pkl
# Uncomment upload_model_and_scaler() to push to S3
```

### 3. Run the Flask API
```bash
cd api
python app.py
# API runs on http://localhost:5000
```

Test the `/predict` endpoint:
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "sequence": [
      {
        "Signal_Strength": "-65 dBm",
        "Latency": "30 ms",
        "Required_Bandwidth": "20 Mbps",
        "Allocated_Bandwidth": "18 Mbps",
        "Application_Type": "Streaming"
      }
      // ... repeat 9 more times (10 total)
    ]
  }'
```

### 4. Run the Gradio UI
```bash
cd ui
python gradio_app.py
# Opens at http://localhost:7860
```

---

## Key Changes from Original

| Issue | Original | Fixed |
|---|---|---|
| LSTM layers | 1-layer hardcoded in app.py | 2-layer matching training, from `config.py` |
| Inference preprocessing | Raw random tensor | MinMaxScaler loaded from `scaler.pkl` |
| Hyperparameters | Scattered, duplicated | Centralized in `config.py` |
| S3 artifacts | Model only | Model **and** scaler |
| REST API | None | `/predict` POST + `/health` GET |
| Model architecture | Redefined in each file | Single `utils/model.py` |

## Additions

| Addition | File |
|---|---|
| EDA (distributions, correlation, class balance) | `notebooks/EDA.py` |
| requirements.txt | `requirements.txt` |
| Model variant comparison (1-layer vs 2-layer vs Linear Regression) | `notebooks/LSTM_Train.py` |
| Per-application-type error breakdown | `notebooks/LSTM_Train.py` |
| Gradio demo UI | `ui/gradio_app.py` |

---

## Technologies
Python · PyTorch · Flask · Gradio · Scikit-learn · Pandas · Matplotlib · Seaborn · AWS EC2 · AWS S3 · Boto3

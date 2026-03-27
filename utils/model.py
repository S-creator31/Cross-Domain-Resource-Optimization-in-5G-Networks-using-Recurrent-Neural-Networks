# utils/model.py
# Defines the LSTM model architecture.
# Imported by BOTH the training notebook and inference code — this is the single
# source of truth for the model structure, eliminating the architecture mismatch bug.

import torch
import torch.nn as nn
from config import MODEL_CONFIG


class LSTMModel(nn.Module):
    """
    2-layer LSTM for 5G resource allocation regression.

    Key fix from original: num_layers=2 and dropout are now pulled from
    MODEL_CONFIG instead of being hardcoded as 1-layer with no dropout.
    Using the same class in training and inference guarantees the saved
    .pth file loads correctly with strict=True (no silent key mismatches).
    """

    def __init__(
        self,
        input_size: int = MODEL_CONFIG["input_size"],
        hidden_size: int = MODEL_CONFIG["hidden_size"],
        num_layers: int = MODEL_CONFIG["num_layers"],
        output_size: int = MODEL_CONFIG["output_size"],
        dropout: float = MODEL_CONFIG["dropout"],
    ):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # dropout only applied between layers, so only matters when num_layers > 1
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=False,  # input shape: (seq_len, batch, input_size)
        )
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()  # output stays in [0, 1] — matches normalized target

    def forward(self, x):
        # x: (seq_len, batch_size, input_size)
        out, (hn, cn) = self.lstm(x)
        # Use the final time-step output for prediction
        out = self.fc(out[-1, :, :])
        out = self.sigmoid(out)
        return out

"""
src/model.py
------------
Deep LSTM Model for RUL Prediction

Architecture:
  Input → LSTM(64) → Dropout(0.2) → LSTM(32) → Dropout(0.2) → Dense(16) → Dense(1)

Why this architecture?
  - Two LSTM layers: first captures short-term patterns, second refines into trend
  - Dropout prevents overfitting (engines vary a lot between themselves)
  - Final Dense(1) outputs a single scalar: predicted RUL value

We use PyTorch (not Keras) for more control over training loop.
"""

import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    """
    Two-layer stacked LSTM for RUL regression.

    Args:
        input_size  : number of features per timestep (e.g., 42)
        hidden_size : number of LSTM hidden units (default: 64)
        num_layers  : number of stacked LSTM layers (default: 2)
        dropout     : dropout probability between layers (default: 0.2)
    """

    def __init__(self, input_size: int, hidden_size: int = 64,
                 num_layers: int = 2, dropout: float = 0.2,
                 bidirectional: bool = True):
        super(LSTMModel, self).__init__()

        self.hidden_size    = hidden_size
        self.num_layers     = num_layers
        self.bidirectional  = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # LSTM stack
        # batch_first=True means input shape: (batch, seq_len, features)
        # CHANGE 1: bidirectional=True -> the LSTM reads each 30-cycle
        # window in BOTH directions (cycle 1->30 AND cycle 30->1) and
        # concatenates both readings at every timestep.
        self.lstm = nn.LSTM(
            input_size    = input_size,
            hidden_size   = hidden_size,
            num_layers    = num_layers,
            batch_first   = True,
            dropout       = dropout if num_layers > 1 else 0.0,
            bidirectional = bidirectional
        )

        # Dropout after LSTM output
        self.dropout = nn.Dropout(dropout)

        # CHANGE 2: fc1's input size doubles when bidirectional=True,
        # because the LSTM now outputs [forward_hidden ; backward_hidden]
        # concatenated at each timestep (hidden_size*2, not hidden_size).
        self.fc1 = nn.Linear(hidden_size * self.num_directions, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)    # Single output = predicted RUL

    def forward(self, x):
        """
        x shape: (batch_size, seq_len, input_size)
        """
        # CHANGE 3: h0/c0's first dimension must be
        # num_layers * num_directions (was just num_layers before),
        # since PyTorch stacks forward+backward hidden states there.
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers * self.num_directions,
                         batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * self.num_directions,
                         batch_size, self.hidden_size).to(x.device)

        # LSTM forward pass
        # out shape: (batch, seq_len, hidden_size * num_directions)
        out, _ = self.lstm(x, (h0, c0))

        # Take only the LAST timestep output (what matters for prediction)
        out = out[:, -1, :]         # shape: (batch, hidden_size)

        out = self.dropout(out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)         # shape: (batch, 1)

        return out.squeeze(-1)      # shape: (batch,)


def count_params(model: nn.Module) -> int:
    """Counts total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---- CMAPSS scoring function (from NASA literature) ----
# This is NOT standard RMSE. PHM literature uses an asymmetric score
# that penalizes late predictions MORE than early predictions.
# Late prediction = predicting MORE remaining life than actually exists → dangerous!

def cmapss_score(y_true, y_pred):
    """
    Official NASA C-MAPSS scoring function.
    d = y_pred - y_true

    d < 0 (predicted less than actual): penalized lightly   → score = e^(-d/13) - 1
    d > 0 (predicted more than actual): penalized heavily   → score = e^(d/10) - 1

    Lower is better. Perfect prediction = 0.
    """
    import numpy as np
    d = y_pred - y_true
    score = np.where(d < 0,
                     np.exp(-d / 13.0) - 1,
                     np.exp(d / 10.0) - 1)
    return float(np.sum(score))


if __name__ == "__main__":
    # Quick sanity check
    model = LSTMModel(input_size=42, hidden_size=64, num_layers=2, dropout=0.2, bidirectional=True)
    print(f"Model architecture:\n{model}")
    print(f"\nTotal trainable parameters: {count_params(model):,}")

    # Test forward pass
    dummy_input = torch.randn(8, 30, 42)   # batch=8, seq=30, features=42
    output = model(dummy_input)
    print(f"\nDummy input shape : {dummy_input.shape}")
    print(f"Output shape       : {output.shape}")    # Should be (8,)
    print(f"Sample predictions : {output.detach().numpy()[:3]}")
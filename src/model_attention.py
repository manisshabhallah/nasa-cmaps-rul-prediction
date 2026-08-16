"""
src/model_attention.py
------------------------
FUTURE WORK: LSTM + Additive Attention for RUL Prediction

Why attention (instead of just "take the last timestep")?
  The vanilla and bidirectional LSTMs both compress the whole 30-cycle
  window into a single representation by either (a) taking only the last
  timestep's hidden state, or (b) blindly reading forward+backward. Neither
  lets the model choose WHICH cycles matter most for this specific window.

  Attention fixes this: instead of always trusting the last cycle equally,
  the model learns a weight for every cycle in the window and takes a
  weighted sum. For FD002/FD004 (multi-condition data), this directly
  targets the problem found in the Bidirectional-LSTM ablation (Slide 07):
  if the operating condition changes partway through a window, attention
  can learn to DOWN-WEIGHT cycles from a different regime instead of
  blindly folding them in via a backward pass.

Architecture:
  Input -> LSTM(64, 2 layers) -> Additive Attention over all 30 timesteps
        -> weighted-sum context vector -> Dense(32) -> Dense(1)

This is a genuine architecture change from src/model.py - swap this file in
for src/model.py (or add as a new option) to try it.

Usage (drop-in, same interface as LSTMModel):
    from src.model_attention import AttentionLSTMModel
    model = AttentionLSTMModel(input_size=42, hidden_size=64, num_layers=2, dropout=0.2)
"""

import torch
import torch.nn as nn


class AdditiveAttention(nn.Module):
    """
    Bahdanau-style additive attention over the LSTM's timestep outputs.

    For each timestep t, computes a scalar "importance score":
        score_t = v^T * tanh(W * h_t)
    Then softmaxes scores across all 30 timesteps to get attention weights,
    and returns the weighted sum of h_t as a single context vector.
    """
    def __init__(self, hidden_size: int):
        super().__init__()
        # W projects each timestep's hidden state before scoring
        self.W = nn.Linear(hidden_size, hidden_size, bias=False)
        # v collapses the projected vector into a single score
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, lstm_out):
        """
        lstm_out shape: (batch, seq_len, hidden_size)
        Returns:
          context: (batch, hidden_size)      - weighted-sum representation
          weights: (batch, seq_len)          - attention weights (for inspection/plotting)
        """
        # scores: (batch, seq_len, 1) -> (batch, seq_len)
        scores = self.v(torch.tanh(self.W(lstm_out))).squeeze(-1)

        # softmax over the time dimension -> each window's weights sum to 1
        weights = torch.softmax(scores, dim=1)          # (batch, seq_len)

        # weighted sum: (batch, seq_len, 1) * (batch, seq_len, hidden) -> sum over seq_len
        context = torch.sum(weights.unsqueeze(-1) * lstm_out, dim=1)   # (batch, hidden)

        return context, weights


class AttentionLSTMModel(nn.Module):
    """
    LSTM + Additive Attention pooling for RUL regression.
    Same constructor signature as the base LSTMModel, so it's a drop-in
    swap in train.py / evaluate.py (just change the import + class name).
    """

    def __init__(self, input_size: int, hidden_size: int = 64,
                 num_layers: int = 2, dropout: float = 0.2,
                 bidirectional: bool = False):
        super().__init__()

        self.hidden_size    = hidden_size
        self.num_layers     = num_layers
        self.bidirectional  = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size    = input_size,
            hidden_size   = hidden_size,
            num_layers    = num_layers,
            batch_first   = True,
            dropout       = dropout if num_layers > 1 else 0.0,
            bidirectional = bidirectional
        )

        # Attention operates over whatever dimension the LSTM outputs
        # (hidden_size, or hidden_size*2 if bidirectional)
        attn_dim = hidden_size * self.num_directions
        self.attention = AdditiveAttention(attn_dim)

        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(attn_dim, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x, return_attention: bool = False):
        """
        x shape: (batch_size, seq_len, input_size)

        Set return_attention=True to also get back the per-timestep
        attention weights - useful for a "what did the model focus on"
        plot in your report (a nice complement to the existing SHAP plots).
        """
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers * self.num_directions,
                         batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * self.num_directions,
                         batch_size, self.hidden_size).to(x.device)

        lstm_out, _ = self.lstm(x, (h0, c0))     # (batch, seq_len, hidden*dirs)

        # Instead of out[:, -1, :] (last timestep only), attention lets the
        # model pick a weighted combination of ALL 30 timesteps.
        context, attn_weights = self.attention(lstm_out)

        out = self.dropout(context)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out).squeeze(-1)

        if return_attention:
            return out, attn_weights
        return out


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Sanity check
    model = AttentionLSTMModel(input_size=42, hidden_size=64, num_layers=2, dropout=0.2)
    print("Model architecture:")
    print(model)
    print(f"Total trainable parameters: {count_params(model):,}")

    x = torch.randn(8, 30, 42)
    out, attn = model(x, return_attention=True)
    print(f"Dummy input shape : {x.shape}")
    print(f"Output shape      : {out.shape}")
    print(f"Attention weights shape: {attn.shape}  (sums to 1 per sample: {attn.sum(dim=1)[:3]})")
import torch
import torch.nn as nn
import math
from app.ml.data.dataset import FEATURE_COLS, SEQUENCE_LEN


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding — injects temporal order into the Transformer."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TransformerEncoder(nn.Module):
    """
    Encoder-only Transformer for financial time-series.

    Input : (batch, seq_len, n_price_features)
    Output: (batch, d_model)  — context-aware representation of the sequence
    """

    def __init__(
        self,
        n_features:  int = len(FEATURE_COLS),
        d_model:     int = 64,
        n_heads:     int = 4,
        n_layers:    int = 2,
        d_ff:        int = 256,
        dropout:     float = 0.1,
        seq_len:     int = SEQUENCE_LEN,
    ):
        super().__init__()
        self.d_model = d_model

        # Project raw features → d_model
        self.input_projection = nn.Linear(n_features, d_model)

        self.pos_encoding = PositionalEncoding(d_model, max_len=seq_len + 10, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,       # (batch, seq, feat) convention
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, n_features)
        x = self.input_projection(x)          # → (batch, seq_len, d_model)
        x = self.pos_encoding(x)
        x = self.transformer(x)               # → (batch, seq_len, d_model)
        x = self.norm(x)
        # Pool: use the last timestep as the sequence representation
        return x[:, -1, :]                    # → (batch, d_model)
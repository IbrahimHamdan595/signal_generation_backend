import torch
import torch.nn as nn
from app.ml.models.transformer import TransformerEncoder
from app.ml.models.mlp_head import MLPHead
from app.ml.data.dataset import FEATURE_COLS, SEQUENCE_LEN
from app.ml.data.entry_time import entry_time_from_bars
from datetime import datetime, timezone


class TradingFusionModel(nn.Module):
    """
    Full multimodal fusion model for trading signal generation.

    Architecture:
        1. TransformerEncoder   → temporal price embedding  Z_T  (d_model,)
        2. Sentiment projection → sentiment embedding       Z_S  (sent_dim,)
        3. Concatenation        → fused vector              Z    (d_model + sent_dim,)
        4. MLPHead              → classification + regression (5 targets)

    Regression targets:
        [0] entry_price
        [1] stop_loss
        [2] take_profit
        [3] net_profit
        [4] bars_to_entry   ← predicted bars until optimal entry (≈1.0)
    """

    def __init__(
        self,
        n_features: int = len(FEATURE_COLS),
        seq_len: int = SEQUENCE_LEN,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 256,
        sent_input: int = 4,
        sent_dim: int = 16,
        mlp_hidden: int = 128,
        dropout: float = 0.1,
        # Alias parameters accepted from saved configs / tests
        price_features: int = None,
        sentiment_features: int = None,
        num_classes: int = None,  # unused — always 3
        reg_targets: int = None,  # unused — always 5
    ):
        # Resolve aliases
        if price_features is not None:
            n_features = price_features
        if sentiment_features is not None:
            sent_input = sentiment_features

        super().__init__()

        self.transformer = TransformerEncoder(
            n_features=n_features,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout,
            seq_len=seq_len,
        )

        self.sentiment_proj = nn.Sequential(
            nn.Linear(sent_input, sent_dim),
            nn.ReLU(),
        )

        fusion_dim = d_model + sent_dim

        self.mlp_head = MLPHead(
            input_dim=fusion_dim,
            hidden_dim=mlp_hidden,
            dropout=dropout,
            n_targets=5,  # entry, sl, tp, net_profit, bars_to_entry
        )

    def forward(self, x_price: torch.Tensor, x_sentiment: torch.Tensor):
        z_t = self.transformer(x_price)  # (batch, d_model)
        z_s = self.sentiment_proj(x_sentiment)  # (batch, sent_dim)
        z = torch.cat([z_t, z_s], dim=-1)  # (batch, fusion_dim)
        return self.mlp_head(z)  # logits(3), regression(5)

    def predict(
        self,
        x_price: torch.Tensor,
        x_sentiment: torch.Tensor,
        current_ts: datetime = None,
        interval: str = "1d",
    ) -> dict:
        """
        Inference helper — returns human-readable predictions including
        the computed entry timestamp based on model-predicted bars_to_entry.
        """
        self.eval()
        with torch.no_grad():
            logits, regression = self.forward(x_price, x_sentiment)

        probs = torch.softmax(logits, dim=-1)
        labels = torch.argmax(probs, dim=-1)

        label_map = {0: "HOLD", 1: "BUY", 2: "SELL"}

        ts_ref = current_ts or datetime.now(timezone.utc)

        bars_to_entry_list = regression[:, 4].tolist()
        entry_times = []
        for bars in bars_to_entry_list:
            entry_ts = entry_time_from_bars(ts_ref, bars, interval)
            entry_times.append(entry_ts.isoformat())

        return {
            "action": [label_map[label.item()] for label in labels],
            "confidence": probs.max(dim=-1).values.tolist(),
            "probabilities": {
                "hold": probs[:, 0].tolist(),
                "buy": probs[:, 1].tolist(),
                "sell": probs[:, 2].tolist(),
            },
            "entry_price": regression[:, 0].tolist(),
            "stop_loss": regression[:, 1].tolist(),
            "take_profit": regression[:, 2].tolist(),
            "net_profit": regression[:, 3].tolist(),
            "bars_to_entry": bars_to_entry_list,
            "entry_time": entry_times,
        }

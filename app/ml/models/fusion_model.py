import torch
import torch.nn as nn
from app.ml.models.transformer import TransformerEncoder
from app.ml.models.mlp_head import MLPHead
from app.ml.models.cross_attention import SentimentCrossAttention
from app.ml.data.dataset import FEATURE_COLS, SEQUENCE_LEN, TIMING_BUCKETS
from app.ml.data.entry_time import entry_time_from_bars
from datetime import datetime, timezone

# Maps discrete timing bucket → bars until entry executes
_BUCKET_TO_BARS = {0: 1.0, 1: 2.0, 2: 3.0, 3: 1.0}  # bucket 3 = HOLD → default next bar


class TradingFusionModel(nn.Module):
    """
    Full multimodal fusion model for trading signal generation.

    Architecture:
        1. TransformerEncoder   → temporal price embedding  Z_T  (d_model,)
        2. Sentiment projection → sentiment embedding       Z_S  (sent_dim,)
        3. Concatenation        → fused vector              Z    (d_model + sent_dim,)
        4. MLPHead              → three heads:
             a. direction logits  (3)    — Hold/Buy/Sell
             b. regression        (5)    — entry_offset_pct, sl, tp, net, bars_to_entry
             c. timing logits     (4)    — entry-bar bucket (bar+1/+2/+3/HOLD)

    All five regression outputs are genuine model predictions:
        [0] entry_offset_pct  — % offset from current close for limit entry
        [1] stop_loss_pct     — signed % SL from entry
        [2] take_profit_pct   — signed % TP from entry
        [3] net_profit_pct    — net reward-risk %
        [4] bars_to_entry     — predicted bars until entry executes (1-3)

    entry_time is deterministically derived from bars_to_entry + calendar.
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
        use_cross_attention: bool = False,
        model_version: str = "v2",   # metadata — ignored at construction
        # Legacy aliases accepted from saved v1 configs
        price_features: int = None,
        sentiment_features: int = None,
        num_classes: int = None,
        reg_targets: int = None,
    ):
        if price_features is not None:
            n_features = price_features
        if sentiment_features is not None:
            sent_input = sentiment_features

        super().__init__()
        self.use_cross_attention = use_cross_attention

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

        if use_cross_attention:
            self.cross_attn = SentimentCrossAttention(
                d_model=d_model,
                sent_dim=sent_dim,
                n_heads=max(1, d_model // 32),   # e.g. d_model=128 → 4 heads
                dropout=dropout,
            )

        fusion_dim = d_model + sent_dim

        self.mlp_head = MLPHead(
            input_dim=fusion_dim,
            hidden_dim=mlp_hidden,
            dropout=dropout,
            n_targets=5,
            n_timing=TIMING_BUCKETS,
        )

    def forward(self, x_price: torch.Tensor, x_sentiment: torch.Tensor):
        z_s = self.sentiment_proj(x_sentiment)    # (batch, sent_dim)

        if self.use_cross_attention:
            # Full sequence → cross-attention: z_t is informed by macro context
            price_seq = self.transformer.forward_seq(x_price)   # (batch, S, d_model)
            z_t       = self.cross_attn(price_seq, z_s)          # (batch, d_model)
        else:
            z_t = self.transformer(x_price)                      # (batch, d_model)

        z = torch.cat([z_t, z_s], dim=-1)         # (batch, fusion_dim)
        return self.mlp_head(z)

    def predict(
        self,
        x_price: torch.Tensor,
        x_sentiment: torch.Tensor,
        current_ts: datetime = None,
        interval: str = "1d",
        temperature: float = 1.0,
    ) -> dict:
        """
        Inference helper — all five regression targets are true model predictions.
        entry_time is derived from predicted bars_to_entry + NYSE calendar.
        """
        self.eval()
        with torch.no_grad():
            dir_logits, regression, timing_logits = self.forward(x_price, x_sentiment)

        dir_probs      = torch.softmax(dir_logits / max(temperature, 0.1), dim=-1)
        timing_probs   = torch.softmax(timing_logits, dim=-1)
        labels         = dir_probs.argmax(dim=-1)
        timing_buckets = timing_probs.argmax(dim=-1)

        label_map = {0: "HOLD", 1: "BUY", 2: "SELL"}
        ts_ref    = current_ts or datetime.now(timezone.utc)

        # Use regression head's continuous bars_to_entry as primary, timing
        # bucket as secondary signal (shown in API for transparency).
        reg_bars = regression[:, 4].tolist()
        entry_times = [
            entry_time_from_bars(ts_ref, max(bars, 1.0), interval).isoformat()
            for bars in reg_bars
        ]

        return {
            "action":     [label_map[lbl.item()] for lbl in labels],
            "confidence": dir_probs.max(dim=-1).values.tolist(),
            "probabilities": {
                "hold": dir_probs[:, 0].tolist(),
                "buy":  dir_probs[:, 1].tolist(),
                "sell": dir_probs[:, 2].tolist(),
            },
            # All five regression outputs — genuine model predictions
            "entry_price":   regression[:, 0].tolist(),  # entry_offset_pct
            "stop_loss":     regression[:, 1].tolist(),  # sl_pct
            "take_profit":   regression[:, 2].tolist(),  # tp_pct
            "net_profit":    regression[:, 3].tolist(),  # net_pct
            "bars_to_entry": reg_bars,                   # continuous
            # Timing head output
            "timing_bucket":      timing_buckets.tolist(),
            "timing_probs":       timing_probs.tolist(),
            "timing_bars":        [_BUCKET_TO_BARS[b.item()] for b in timing_buckets],
            # Derived
            "entry_time": entry_times,
        }

    def mc_predict(
        self,
        x_price: torch.Tensor,
        x_sentiment: torch.Tensor,
        n_samples: int = 20,
        temperature: float = 1.0,
        current_ts: datetime = None,
        interval: str = "1d",
    ) -> dict:
        """Monte Carlo Dropout inference: N forward passes with dropout active.

        Confidence = mean(max_prob) − 0.5 × std(max_prob_at_predicted_class)

        Penalises predictions that are individually confident but inconsistent
        across passes — a much more reliable quality signal than raw softmax.
        Falls back to standard predict() when n_samples <= 1.
        """
        if n_samples <= 1:
            return self.predict(x_price, x_sentiment, current_ts, interval, temperature)

        # Enable dropout by setting train mode, but disable gradient computation
        self.train()

        dir_probs_list:   list = []
        regression_list:  list = []
        timing_probs_list: list = []

        with torch.no_grad():
            for _ in range(n_samples):
                dl, reg, tl = self.forward(x_price, x_sentiment)
                dir_probs_list.append(torch.softmax(dl / max(temperature, 0.1), dim=-1))
                regression_list.append(reg)
                timing_probs_list.append(torch.softmax(tl, dim=-1))

        self.eval()

        # Stack: (n_samples, batch, classes)
        dp  = torch.stack(dir_probs_list,   dim=0)
        reg = torch.stack(regression_list,  dim=0)
        tp  = torch.stack(timing_probs_list, dim=0)

        mean_dp  = dp.mean(dim=0)    # (batch, 3)
        std_dp   = dp.std(dim=0)     # (batch, 3)
        mean_reg = reg.mean(dim=0)   # (batch, 5)
        mean_tp  = tp.mean(dim=0)    # (batch, 4)

        labels        = mean_dp.argmax(dim=-1)
        mean_max_prob = mean_dp.max(dim=-1).values
        # std at the predicted class — high std = model disagrees across passes
        std_at_pred   = std_dp.gather(1, labels.unsqueeze(1)).squeeze(1)
        confidence    = (mean_max_prob - 0.5 * std_at_pred).clamp(0.0, 1.0)

        timing_buckets = mean_tp.argmax(dim=-1)
        label_map      = {0: "HOLD", 1: "BUY", 2: "SELL"}
        ts_ref         = current_ts or datetime.now(timezone.utc)
        reg_bars       = mean_reg[:, 4].tolist()

        entry_times = [
            entry_time_from_bars(ts_ref, max(bars, 1.0), interval).isoformat()
            for bars in reg_bars
        ]

        return {
            "action":      [label_map[lbl.item()] for lbl in labels],
            "confidence":  confidence.tolist(),
            "uncertainty": std_at_pred.tolist(),
            "probabilities": {
                "hold": mean_dp[:, 0].tolist(),
                "buy":  mean_dp[:, 1].tolist(),
                "sell": mean_dp[:, 2].tolist(),
            },
            "entry_price":   mean_reg[:, 0].tolist(),
            "stop_loss":     mean_reg[:, 1].tolist(),
            "take_profit":   mean_reg[:, 2].tolist(),
            "net_profit":    mean_reg[:, 3].tolist(),
            "bars_to_entry": reg_bars,
            "timing_bucket":      timing_buckets.tolist(),
            "timing_probs":       mean_tp.tolist(),
            "timing_bars":        [_BUCKET_TO_BARS[b.item()] for b in timing_buckets],
            "entry_time":         entry_times,
        }

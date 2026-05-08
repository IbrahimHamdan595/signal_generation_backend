"""SentimentCrossAttention — macro/sentiment context attends into the price sequence.

Instead of simply concatenating a pooled price vector with a sentiment scalar,
this module lets the sentiment/macro vector query the full 60-bar price sequence
so the model learns *which historical bars* are most relevant given today's
macro context.

Example: "FOMC is tomorrow + compound sentiment = -0.4"
  → the model learns to up-weight bars from previous FOMC windows in the price
    sequence, producing a context-aware price representation.

Architecture:
    Q  ← Linear(sent_dim  → d_model)   applied to sentiment  (batch, 1, d_model)
    K  ← Linear(d_model   → d_model)   applied to price seq  (batch, S, d_model)
    V  ← Linear(d_model   → d_model)   applied to price seq  (batch, S, d_model)

    attn = softmax(Q Kᵀ / √d_head)     (batch, n_heads, 1, S)
    out  = attn · V                     → (batch, d_model) after head merging

    Residual: out = LayerNorm(out + mean_pool(price_seq))
    The residual ensures gradient flow even when attention is diffuse and
    preserves the mean-pool baseline that already works well.
"""

import math
import torch
import torch.nn as nn


class SentimentCrossAttention(nn.Module):
    def __init__(
        self,
        d_model:  int,
        sent_dim: int,
        n_heads:  int   = 4,
        dropout:  float = 0.1,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads
        self.d_model = d_model

        self.q_proj   = nn.Linear(sent_dim, d_model)
        self.k_proj   = nn.Linear(d_model,  d_model)
        self.v_proj   = nn.Linear(d_model,  d_model)
        self.out_proj = nn.Linear(d_model,  d_model)
        self.dropout  = nn.Dropout(dropout)
        self.norm     = nn.LayerNorm(d_model)

    def forward(self, price_seq: torch.Tensor, sentiment: torch.Tensor) -> torch.Tensor:
        """
        price_seq : (batch, seq_len, d_model)
        sentiment : (batch, sent_dim)
        Returns   : (batch, d_model)
        """
        B, S, _ = price_seq.shape

        # Query: sentiment vector → (batch, 1, d_model)
        Q = self.q_proj(sentiment).unsqueeze(1)
        K = self.k_proj(price_seq)   # (batch, S, d_model)
        V = self.v_proj(price_seq)   # (batch, S, d_model)

        def _split(t: torch.Tensor, seq: int) -> torch.Tensor:
            # (batch, seq, d_model) → (batch, n_heads, seq, d_head)
            return t.view(B, seq, self.n_heads, self.d_head).transpose(1, 2)

        Q = _split(Q, 1)    # (batch, n_heads, 1, d_head)
        K = _split(K, S)    # (batch, n_heads, S, d_head)
        V = _split(V, S)    # (batch, n_heads, S, d_head)

        scale  = math.sqrt(self.d_head)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / scale  # (B, H, 1, S)
        scores = torch.clamp(scores, min=-30.0, max=30.0)       # prevent softmax NaN from extreme init
        attn   = torch.softmax(scores, dim=-1)
        attn   = self.dropout(attn)

        # (B, H, 1, d_head) → (B, d_model)
        out = torch.matmul(attn, V)                      # (B, H, 1, d_head)
        out = out.transpose(1, 2).contiguous()            # (B, 1, H, d_head)
        out = out.view(B, self.d_model)                   # (B, d_model)
        out = self.out_proj(out)

        # Residual with mean-pooled price sequence: keeps the mean-pool
        # baseline alive and stabilises early training
        mean_price = price_seq.mean(dim=1)               # (B, d_model)
        return self.norm(out + mean_price)

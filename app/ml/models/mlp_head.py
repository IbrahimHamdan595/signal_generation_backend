import torch
import torch.nn as nn


class MLPHead(nn.Module):
    """
    Triple-output MLP prediction head.

    Input : fused embedding (batch, fusion_dim)
    Outputs:
        logits         : (batch, 3)  — Hold/Buy/Sell direction
        regression     : (batch, 5)  — entry_offset_pct, sl, tp, net, bars_to_entry
        timing_logits  : (batch, 4)  — entry timing bucket (bar+1/+2/+3/HOLD)
    """

    def __init__(
        self,
        input_dim:  int,
        hidden_dim: int   = 128,
        dropout:    float = 0.2,
        n_classes:  int   = 3,    # Hold=0, Buy=1, Sell=2
        n_targets:  int   = 5,    # entry_offset, sl, tp, net, bars_to_entry
        n_timing:   int   = 4,    # timing buckets: bar+1, +2, +3, HOLD/no-fill
    ):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Direction — raw logits (CrossEntropyLoss handles softmax)
        self.classifier = nn.Linear(hidden_dim // 2, n_classes)

        # Regression — continuous price/timing targets
        self.regressor  = nn.Linear(hidden_dim // 2, n_targets)

        # Timing — discrete entry-bar classification (small separate head)
        self.timing_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, n_timing),
        )

    def forward(self, x: torch.Tensor):
        h = self.shared(x)
        return self.classifier(h), self.regressor(h), self.timing_head(h)
import torch
import torch.nn as nn


class MLPHead(nn.Module):
    """
    Dual-output MLP prediction head.

    Input : fused embedding (batch, fusion_dim)
    Output:
        logits     : (batch, 3)  — Buy / Sell / Hold classification
        regression : (batch, 5)  — Entry Price, Stop Loss, Take Profit,
                                   Net Profit, Bars to Entry
    """

    def __init__(
        self,
        input_dim:  int,
        hidden_dim: int   = 128,
        dropout:    float = 0.2,
        n_classes:  int   = 3,    # Buy=1, Sell=2, Hold=0
        n_targets:  int   = 5,    # entry, sl, tp, net_profit, bars_to_entry
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

        # Classification — raw logits (CrossEntropyLoss handles softmax)
        self.classifier = nn.Linear(hidden_dim // 2, n_classes)

        # Regression — linear for continuous targets
        self.regressor  = nn.Linear(hidden_dim // 2, n_targets)

    def forward(self, x: torch.Tensor):
        h = self.shared(x)
        return self.classifier(h), self.regressor(h)
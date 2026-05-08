"""Temperature scaling — post-hoc confidence calibration for the direction head.

After training completes, a single scalar T is fitted on the validation set by
minimising the Negative Log-Likelihood (NLL) of the direction predictions.

    calibrated_confidence = softmax(logits / T).max()

T > 1  →  softens probabilities  (fixes overconfidence — the typical case)
T < 1  →  sharpens probabilities (fixes underconfidence)
T = 1  →  no change (raw softmax)

Fitting is cheap: one pass over the val set to collect logits, then LBFGS
optimisation over a single scalar.  No gradient through the model.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import logging

logger = logging.getLogger(__name__)

_T_MIN = 0.5
_T_MAX = 5.0


class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))

    def fit(self, model: nn.Module, val_loader: DataLoader) -> float:
        """Collect val-set logits then optimise T.  Returns the optimal T value."""
        device = next(model.parameters()).device
        model.eval()

        all_logits: list = []
        all_labels: list = []

        with torch.no_grad():
            for batch in val_loader:
                x_price, x_sent, y_cls = batch[0], batch[1], batch[2]
                dir_logits, _, _ = model(
                    x_price.to(device), x_sent.to(device)
                )
                all_logits.append(dir_logits.cpu())
                all_labels.append(y_cls.cpu())

        logits = torch.cat(all_logits)   # (N, 3)
        labels = torch.cat(all_labels)   # (N,)

        self.cpu()
        nll = nn.CrossEntropyLoss()

        optimizer = torch.optim.LBFGS(
            [self.temperature], lr=0.05, max_iter=200, tolerance_grad=1e-7
        )

        def _step():
            optimizer.zero_grad()
            T     = self.temperature.clamp(_T_MIN, _T_MAX)
            loss  = nll(logits / T, labels)
            loss.backward()
            return loss

        optimizer.step(_step)

        T = float(self.temperature.clamp(_T_MIN, _T_MAX).item())
        logger.info(f"🌡️  Temperature calibration complete: T = {T:.4f}")
        return T

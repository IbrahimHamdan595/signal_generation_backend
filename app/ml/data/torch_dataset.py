import torch
from torch.utils.data import Dataset
import numpy as np


class TradingDataset(Dataset):
    """
    PyTorch Dataset wrapping the numpy arrays produced by DatasetBuilder.

    Each sample:
        x_price     : (seq_len, n_price_features)  float32
        x_sentiment : (4,)                          float32
        y_class     : scalar int64  — 0=Hold, 1=Buy, 2=Sell
        y_regression: (5,)          float32 — [entry, sl, tp, net_profit, max_dd]
    """

    def __init__(
        self,
        X_price:      np.ndarray,   # (N, seq_len, F)
        X_sentiment:  np.ndarray,   # (N, 4)
        y_class:      np.ndarray,   # (N,)
        y_regression: np.ndarray,   # (N, 4)
    ):
        self.X_price      = torch.tensor(X_price,      dtype=torch.float32)
        self.X_sentiment  = torch.tensor(X_sentiment,  dtype=torch.float32)
        self.y_class      = torch.tensor(y_class,      dtype=torch.long)
        self.y_regression = torch.tensor(y_regression, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.y_class)

    def __getitem__(self, idx):
        return (
            self.X_price[idx],
            self.X_sentiment[idx],
            self.y_class[idx],
            self.y_regression[idx],
        )


def split_dataset(
    X_price:      np.ndarray,
    X_sentiment:  np.ndarray,
    y_class:      np.ndarray,
    y_regression: np.ndarray,
    train_ratio:  float = 0.70,
    val_ratio:    float = 0.15,
) -> tuple:
    """
    Chronological split — NO shuffling to avoid look-ahead bias.
    Returns (train_ds, val_ds, test_ds)
    """
    N = len(y_class)
    t1 = int(N * train_ratio)
    t2 = int(N * (train_ratio + val_ratio))

    def make(s, e):
        return TradingDataset(
            X_price[s:e], X_sentiment[s:e],
            y_class[s:e], y_regression[s:e],
        )

    return make(0, t1), make(t1, t2), make(t2, N)
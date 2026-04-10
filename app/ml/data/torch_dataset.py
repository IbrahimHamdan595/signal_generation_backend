import torch
from torch.utils.data import Dataset, ConcatDataset
import numpy as np
from typing import List


class TradingDataset(Dataset):
    """
    PyTorch Dataset wrapping the numpy arrays produced by DatasetBuilder.

    Each sample:
        x_price     : (seq_len, n_price_features)  float32
        x_sentiment : (4,)                          float32
        y_class     : scalar int64  — 0=Hold, 1=Buy, 2=Sell
        y_regression: (5,)          float32 — [entry, sl, tp, net_profit, bars_to_entry]
    """

    def __init__(
        self,
        X_price:      np.ndarray,   # (N, seq_len, F)
        X_sentiment:  np.ndarray,   # (N, 4)
        y_class:      np.ndarray,   # (N,)
        y_regression: np.ndarray,   # (N, 5)
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
    Global chronological split — kept for compatibility with walk-forward
    and cases where the caller has already done per-ticker splitting.
    NO shuffling to avoid look-ahead bias.
    Returns (train_ds, val_ds, test_ds).
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


def split_dataset_per_ticker(
    ticker_data: List[tuple],
    train_ratio: float = 0.70,
    val_ratio:   float = 0.15,
) -> tuple:
    """
    Ticker-level chronological split that eliminates data leakage.

    For each ticker, we split its own time-series into train/val/test
    proportions independently, then concatenate across tickers.
    This ensures that no future bar from one ticker contaminates another
    ticker's training window — even if they traded on the same calendar date.

    Args:
        ticker_data: list of (X_price, X_sent, y_cls, y_reg) tuples,
                     one per ticker, in chronological order within each tuple.
        train_ratio: fraction of each ticker's bars used for training.
        val_ratio:   fraction used for validation.

    Returns:
        (train_ds, val_ds, test_ds)  — ConcatDataset over all tickers.
    """
    train_parts, val_parts, test_parts = [], [], []

    for (xp, xs, yc, yr) in ticker_data:
        N  = len(yc)
        if N < 10:          # too few samples — skip
            continue
        t1 = int(N * train_ratio)
        t2 = int(N * (train_ratio + val_ratio))

        def _ds(s, e):
            return TradingDataset(xp[s:e], xs[s:e], yc[s:e], yr[s:e])

        train_parts.append(_ds(0,  t1))
        val_parts.append(  _ds(t1, t2))
        test_parts.append( _ds(t2, N))

    if not train_parts:
        raise ValueError("No ticker produced enough samples for splitting.")

    return (
        ConcatDataset(train_parts),
        ConcatDataset(val_parts),
        ConcatDataset(test_parts),
    )

import torch
from torch.utils.data import Dataset, ConcatDataset
import numpy as np
from typing import List


class TradingDataset(Dataset):
    """
    PyTorch Dataset wrapping the numpy arrays produced by DatasetBuilder.

    Each sample (5-tuple):
        x_price     : (seq_len, n_price_features)  float32
        x_sentiment : (4,)                          float32
        y_class     : scalar int64  — 0=Hold, 1=Buy, 2=Sell
        y_regression: (5,) float32  — [entry_offset_pct, sl, tp, net, bars_to_entry]
        y_timing    : scalar int64  — 0=bar+1, 1=bar+2, 2=bar+3, 3=HOLD/no-fill
    """

    def __init__(
        self,
        X_price:      np.ndarray,   # (N, seq_len, F)
        X_sentiment:  np.ndarray,   # (N, 4)
        y_class:      np.ndarray,   # (N,)
        y_regression: np.ndarray,   # (N, 5)
        y_timing:     np.ndarray,   # (N,)
    ):
        self.X_price      = torch.tensor(X_price,      dtype=torch.float32)
        self.X_sentiment  = torch.tensor(X_sentiment,  dtype=torch.float32)
        self.y_class      = torch.tensor(y_class,      dtype=torch.long)
        self.y_regression = torch.tensor(y_regression, dtype=torch.float32)
        self.y_timing     = torch.tensor(y_timing,     dtype=torch.long)

    def __len__(self) -> int:
        return len(self.y_class)

    def __getitem__(self, idx):
        return (
            self.X_price[idx],
            self.X_sentiment[idx],
            self.y_class[idx],
            self.y_regression[idx],
            self.y_timing[idx],
        )


def split_dataset(
    X_price:      np.ndarray,
    X_sentiment:  np.ndarray,
    y_class:      np.ndarray,
    y_regression: np.ndarray,
    y_timing:     np.ndarray,
    train_ratio:  float = 0.70,
    val_ratio:    float = 0.15,
) -> tuple:
    """
    Global chronological split — kept for walk-forward and cases where the
    caller has already done per-ticker splitting. NO shuffling.
    Returns (train_ds, val_ds, test_ds).
    """
    N = len(y_class)
    t1 = int(N * train_ratio)
    t2 = int(N * (train_ratio + val_ratio))

    def make(s, e):
        return TradingDataset(
            X_price[s:e], X_sentiment[s:e],
            y_class[s:e], y_regression[s:e], y_timing[s:e],
        )

    return make(0, t1), make(t1, t2), make(t2, N)


def split_dataset_per_ticker(
    ticker_data: List[tuple],
    train_ratio: float = 0.70,
    val_ratio:   float = 0.15,
    stagger:     float = 0.05,
    seed:        int   = 42,
) -> tuple:
    """
    Ticker-level chronological split that eliminates per-ticker leakage AND
    de-correlates the test set from a single market regime.

    For each ticker, we split its own time-series chronologically, then
    concatenate across tickers. To avoid every ticker's test slice landing
    in the same calendar window (which made test accuracy a regime-specific
    measurement), we add a small deterministic per-ticker jitter to the
    split boundaries. Each ticker still trains on its own past and tests on
    its own future — only the position of that boundary varies.

    Args:
        ticker_data: list of (X_price, X_sent, y_cls, y_reg) tuples,
                     one per ticker, in chronological order within each tuple.
        train_ratio: target fraction of each ticker's bars used for training.
        val_ratio:   target fraction used for validation.
        stagger:     half-width of the uniform jitter applied to train_ratio
                     (default ±5 percentage points). Set to 0 to disable.
        seed:        seed for the jitter so splits are reproducible.

    Returns:
        (train_ds, val_ds, test_ds)  — ConcatDataset over all tickers.
    """
    rng = np.random.default_rng(seed)
    train_parts, val_parts, test_parts = [], [], []

    for (xp, xs, yc, yr, yt) in ticker_data:
        N  = len(yc)
        if N < 10:          # too few samples — skip
            continue

        if stagger > 0:
            jitter = float(rng.uniform(-stagger, stagger))
            tr = max(0.50, min(0.85, train_ratio + jitter))
        else:
            tr = train_ratio

        t1 = int(N * tr)
        t2 = int(N * (tr + val_ratio))
        t1 = min(max(t1, 1), N - 2)
        t2 = min(max(t2, t1 + 1), N - 1)

        def _ds(s, e):
            return TradingDataset(xp[s:e], xs[s:e], yc[s:e], yr[s:e], yt[s:e])

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

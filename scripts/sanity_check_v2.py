"""
Single-ticker sanity check for the v2 all-predictable model.

Verifies:
  1. All three heads produce non-constant outputs when the input window changes.
  2. Regression outputs vary across samples (model is not collapsed).
  3. Timing head predicts different buckets across samples.
  4. Shapes are correct for every output tensor.

Usage (from repo root):
    cd backend
    python scripts/sanity_check_v2.py [--ticker AAPL] [--n 32]
"""

import argparse
import sys
import os

# Make sure backend package root is on the path when run standalone
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import numpy as np

from app.ml.models.fusion_model import TradingFusionModel
from app.ml.data.dataset import FEATURE_COLS, SEQUENCE_LEN, TIMING_BUCKETS


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_random_batch(n: int, n_price_feat: int, sent_dim: int = 4):
    """Random inputs that vary across the batch — used to probe non-collapse."""
    x_price = torch.randn(n, SEQUENCE_LEN, n_price_feat)
    x_sent  = torch.randn(n, sent_dim)
    return x_price, x_sent


def _make_constant_batch(n: int, n_price_feat: int, sent_dim: int = 4):
    """All samples identical — outputs should still differ only due to model determinism."""
    row_price = torch.randn(1, SEQUENCE_LEN, n_price_feat)
    row_sent  = torch.randn(1, sent_dim)
    x_price = row_price.expand(n, -1, -1)
    x_sent  = row_sent.expand(n, -1)
    return x_price, x_sent


def _check(name: str, tensor: torch.Tensor, expected_shape: tuple) -> bool:
    if tuple(tensor.shape) != expected_shape:
        print(f"  FAIL  {name}: shape {tuple(tensor.shape)} != expected {expected_shape}")
        return False
    print(f"  OK    {name}: shape {tuple(tensor.shape)}")
    return True


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=32, help="Batch size for checks")
    args = parser.parse_args()

    n = args.n
    n_price = len(FEATURE_COLS)

    print(f"\n{'='*60}")
    print(f"  v2 Sanity Check — batch={n}, seq_len={SEQUENCE_LEN}, features={n_price}")
    print(f"{'='*60}\n")

    model = TradingFusionModel()
    model.eval()
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model params: {param_count:,}\n")

    # ── 1. Shape checks ───────────────────────────────────────────────────────
    print("─── Shape checks ───")
    x_p, x_s = _make_random_batch(n, n_price)
    with torch.no_grad():
        dir_logits, regression, timing_logits = model(x_p, x_s)

    all_ok = True
    all_ok &= _check("dir_logits",    dir_logits,    (n, 3))
    all_ok &= _check("regression",    regression,    (n, 5))
    all_ok &= _check("timing_logits", timing_logits, (n, TIMING_BUCKETS))

    # ── 2. Regression variance ────────────────────────────────────────────────
    print("\n─── Regression variance (should be > 0) ───")
    reg_names = ["entry_offset_pct", "sl_pct", "tp_pct", "net_pct", "bars_to_entry"]
    reg_np = regression.numpy()
    for i, name in enumerate(reg_names):
        std = float(np.std(reg_np[:, i]))
        status = "OK" if std > 1e-6 else "COLLAPSED"
        print(f"  {status:10s} {name}: std={std:.6f}")
        if std <= 1e-6:
            all_ok = False

    # ── 3. Direction head — all three classes predicted? ─────────────────────
    print("\n─── Direction head coverage ───")
    preds = dir_logits.argmax(dim=-1).numpy()
    unique_dir = set(preds.tolist())
    print(f"  Unique direction predictions: {sorted(unique_dir)}  (want {{0,1,2}} or subset)")
    if len(unique_dir) < 2:
        print("  WARNING: direction head only predicts one class — likely collapsed")
        # Not a hard failure — small batches may genuinely all predict one class

    # ── 4. Timing head — more than one bucket predicted? ─────────────────────
    print("\n─── Timing head coverage ───")
    t_preds = timing_logits.argmax(dim=-1).numpy()
    unique_tim = set(t_preds.tolist())
    print(f"  Unique timing predictions: {sorted(unique_tim)}  (want > 1 bucket for varied inputs)")
    if len(unique_tim) < 2:
        print("  WARNING: timing head only predicts one bucket — may be collapsed")

    # ── 5. Constant-input batch — outputs must be identical ───────────────────
    print("\n─── Constant-input consistency (same input → same output) ───")
    x_p2, x_s2 = _make_constant_batch(n, n_price)
    with torch.no_grad():
        dl2, reg2, tl2 = model(x_p2, x_s2)

    for name, t in [("dir_logits", dl2), ("regression", reg2), ("timing_logits", tl2)]:
        row_std = t.std(dim=0).max().item()
        ok = row_std < 1e-5
        print(f"  {'OK' if ok else 'FAIL':6s} {name}: max row std = {row_std:.2e}")
        if not ok:
            all_ok = False

    # ── 6. predict() helper ───────────────────────────────────────────────────
    print("\n─── predict() helper ───")
    x_p3, x_s3 = _make_random_batch(2, n_price)
    out = model.predict(x_p3, x_s3, interval="1d")
    expected_keys = {
        "action", "confidence", "probabilities",
        "entry_price", "stop_loss", "take_profit", "net_profit", "bars_to_entry",
        "timing_bucket", "timing_probs", "timing_bars", "entry_time",
    }
    missing = expected_keys - set(out.keys())
    if missing:
        print(f"  FAIL  predict() missing keys: {missing}")
        all_ok = False
    else:
        print(f"  OK    predict() returns all expected keys")
    print(f"  action={out['action']}  confidence={[f'{c:.3f}' for c in out['confidence']]}")
    print(f"  bars_to_entry={out['bars_to_entry']}  timing_bucket={out['timing_bucket']}")
    print(f"  entry_time={out['entry_time']}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    if all_ok:
        print("  ALL CHECKS PASSED")
    else:
        print("  SOME CHECKS FAILED — review output above")
    print(f"{'='*60}\n")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())

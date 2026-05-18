"""One-shot FX training kick-off.

Trains the FX TradingFusionModel on the 17 yfinance-covered pairs (3 MT5-only
pairs are skipped automatically by `_quality_filter` if they have no bars).
Uses FX-tuned labelling thresholds: ±1.2% barriers over a 10-bar lookahead.

Run as a background process; tail the log to monitor:
    cd backend
    .venv/Scripts/python scripts/train_fx.py > checkpoints/fx/training.log 2>&1 &

The trained model is registered under `checkpoints/fx/` and immediately
available for inference via the multi-model registry.
"""

from __future__ import annotations

import asyncio
import os
import sys

# Allow direct script execution from anywhere
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


async def main() -> int:
    from app.db.database import connect_db, get_db, close_db
    from app.core.asset_class import FX_TICKERS
    from app.services.ml_service import MLService

    await connect_db()
    pool = await get_db()

    tickers = sorted(FX_TICKERS)
    print(f"[train_fx] starting FX training on {len(tickers)} pairs")
    print(f"[train_fx] tickers: {tickers}")

    svc = MLService(pool)
    # Tier 1+2+3 config:
    #   - barrier_atr_mult=1.5: vol-normalised labels (1.5×ATR barriers per
    #     pair per bar) — every pair gets the same label-density distribution
    #     regardless of its native daily volatility. Textbook FX-ML fix.
    #   - lookahead_window=5: short enough that barrier hits are decisive,
    #     long enough for trends to develop.
    #   - use_class_weights=False: with vol-normalised labels, class
    #     distribution should already be more balanced — let it flow.
    #   - 40 epochs: training F1 was still climbing at epoch 30 last run.
    #   - Tier 2 features (us_2y_yield, realized_vol_20d, vol_of_vol_5d) are
    #     added automatically by the dataset builder for FX tickers; the
    #     model_config will record n_features=57.
    result = await svc.train(
        tickers=tickers,
        interval="1d",
        epochs=40,
        batch_size=64,
        lr=2e-4,
        use_class_weights=False,
        diagnostics=False,
        asset_class="fx",
        # Empirically-tuned for balanced class distribution on FX majors:
        # 1.5×ATR over 10 bars gives ~36% HOLD / 33% BUY / 31% SELL on EURUSD —
        # no class dominates and BUY/SELL are roughly symmetric so the model
        # can't take a directional shortcut.
        barrier_atr_mult=1.5,
        # Fallbacks for bars where ATR is missing/zero (rare on FX majors)
        buy_threshold=0.008,
        sell_threshold=0.008,
        lookahead_window=10,
    )

    # Persist last_train_result.json under checkpoints/fx/
    import json
    out = os.path.join("checkpoints", "fx", "last_train_result.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=2, default=str)

    print(f"\n[train_fx] DONE")
    print(f"  val_loss:     {result.get('training', {}).get('best_val_loss')}")
    print(f"  test_acc:     {result.get('evaluation', {}).get('accuracy')}")
    print(f"  test_sharpe:  {result.get('evaluation', {}).get('trading', {}).get('sharpe_ratio')}")
    print(f"  win_rate:     {result.get('evaluation', {}).get('trading', {}).get('win_rate')}")
    print(f"  class_dist:   {result.get('dataset', {}).get('class_dist')}")

    await close_db()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

"""Re-tune confidence + margin thresholds for the existing FX checkpoint.

Loads the trained FX model, rebuilds the validation slice (same split as
training), runs the patched threshold sweep, and writes the new
`confidence_threshold` / `margin_threshold` (plus `disabled_actions=["SELL"]`)
into `checkpoints/fx/model_config.json`.

Used after Tier 1+2+3 training to fix:
  - the buggy strict-precision floor that returned hardcoded defaults
  - the SELL anti-edge (35% precision -> downgrade to HOLD at inference)

Usage:
    cd backend
    .venv/Scripts/python scripts/retune_fx.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


async def main() -> int:
    import torch
    from torch.utils.data import DataLoader
    import numpy as np

    from app.db.database import connect_db, get_db, close_db
    from app.core.asset_class import FX_TICKERS
    from app.services.ml_service import MLService
    from app.ml.models.registry import (
        get_model, _paths_for, save_model_config,
    )
    from app.ml.data.dataset import DatasetBuilder, SEQUENCE_LEN
    from app.ml.data.torch_dataset import split_dataset_per_ticker

    await connect_db()
    pool = await get_db()
    svc = MLService(pool)

    model = get_model("fx")
    if model is None:
        print("[error] no FX model loaded — train first")
        return 1

    # Rebuild the val split the same way training did, with the same overrides
    tickers = sorted(FX_TICKERS)
    print(f"[retune] rebuilding val split for {len(tickers)} FX tickers...")
    builder = DatasetBuilder(pool)
    ticker_data: list = []
    for t in tickers:
        cached = await svc._load_ticker_cache(
            t, "1d", SEQUENCE_LEN,
            barrier_atr_mult=1.5, buy_thresh=0.008, sell_thresh=0.008, lookahead=10,
        )
        if cached is None:
            print(f"  [{t}] no cached dataset — skipping (mismatched params)")
            continue
        p, s, c, r, tt = cached
        # Normalisation: load the same per-ticker scaler the model was trained with
        # (already saved in checkpoints/fx/scaler_params.json)
        ticker_data.append((p, s, c, r, tt))
        print(f"  [{t}] {len(c)} sequences")

    if not ticker_data:
        print("[error] no cached datasets found — re-run train_fx.py first")
        return 1

    # Apply the saved per-ticker scalers (same as training did)
    from app.ml.models.registry import load_scaler_params
    scalers = load_scaler_params("fx") or {}
    available_tickers = [t for t in tickers if any(True for _ in ticker_data)]
    # Rebuild the (ticker, arrays) pairing — ticker_data items align with
    # the order tickers were iterated above and successfully loaded
    pair_idx = 0
    normalised = []
    for t in tickers:
        if pair_idx >= len(ticker_data):
            break
        # We loaded ticker_data in the same order as `tickers`, skipping
        # ones without cache. We need to figure out which ticker each
        # ticker_data[i] corresponds to. Rebuild the alignment.
        pair_idx += 1
    # Simpler: re-do the loading loop together with normalisation
    normalised = []
    for t in tickers:
        cached = await svc._load_ticker_cache(
            t, "1d", SEQUENCE_LEN,
            barrier_atr_mult=1.5, buy_thresh=0.008, sell_thresh=0.008, lookahead=10,
        )
        if cached is None:
            continue
        p, s, c, r, tt = cached
        sc = scalers.get(t) or scalers.get(t.upper())
        if sc:
            mean = np.array(sc["mean"], dtype=np.float32)
            std  = np.maximum(np.array(sc["std"], dtype=np.float32), 1e-2)
            p_norm = np.clip((p - mean) / std, -10.0, 10.0).astype(np.float32)
        else:
            p_norm = p
        normalised.append((p_norm, s, c, r, tt))

    _, val_ds, _ = split_dataset_per_ticker(normalised)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, drop_last=False)
    print(f"[retune] val_ds size: {len(val_ds)}")

    # Read current temperature (could be NaN) - the sweep falls back to 1.0
    paths = _paths_for("fx")
    with open(paths["config"]) as f:
        cfg = json.load(f)
    temp_raw = cfg.get("temperature", 1.0)
    try:
        temp = float(temp_raw)
        if not np.isfinite(temp) or temp <= 0:
            temp = 1.0
    except Exception:
        temp = 1.0

    print(f"[retune] running sweep with temperature={temp}, asset_class='fx'")
    tuned = svc._tune_thresholds(model, val_loader, temperature=temp, asset_class="fx")
    print(f"[retune] best combo: {tuned}")

    # Patch model_config.json with the new thresholds + disable SELLs
    cfg["confidence_threshold"] = tuned["confidence_threshold"]
    cfg["margin_threshold"]     = tuned["margin_threshold"]
    cfg["disabled_actions"]     = ["SELL"]
    cfg["temperature"]          = temp   # write back the sanitised value
    save_model_config(cfg, asset_class="fx")
    print(f"[retune] DONE - model_config.json updated:")
    print(f"  confidence_threshold = {cfg['confidence_threshold']}")
    print(f"  margin_threshold     = {cfg['margin_threshold']}")
    print(f"  disabled_actions     = {cfg['disabled_actions']}")
    print(f"  temperature          = {cfg['temperature']}")

    await close_db()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

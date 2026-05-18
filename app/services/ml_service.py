import os
import torch
import numpy as np
import logging
import asyncpg
from torch.utils.data import DataLoader
from datetime import datetime, timezone
from typing import List, Optional

from app.ml.data.dataset import DatasetBuilder, FEATURE_COLS, SEQUENCE_LEN
from app.ml.data.entry_time import entry_time_from_bars
from app.ml.data.torch_dataset import split_dataset_per_ticker
from app.ml.models.fusion_model import TradingFusionModel
from app.ml.models.registry import (
    save_model_config,
    save_scaler_params,
    register_version,
    new_checkpoint_name,
    reload_model,
)
from app.ml.training.trainer import Trainer, compute_class_weights
from app.ml.evaluation.evaluator import ModelEvaluator
from app.core.config import settings

logger = logging.getLogger(__name__)


_DATASET_CACHE_DIR = "checkpoints/dataset_cache"


class MLService:
    def __init__(self, pool: asyncpg.Pool):
        self.pool = pool

    # ── Per-ticker dataset cache ──────────────────────────────────────────────
    # Stores the build outputs of DatasetBuilder._build_ticker keyed by
    # (ticker, interval, seq_len, latest_bar_timestamp). When the next train
    # job runs and no new bar has arrived, we reuse the cached numpy arrays
    # instead of re-running per-ticker SQL + sentiment + feature joins.

    async def _latest_bar_ts(self, ticker: str, interval: str) -> Optional[str]:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT timestamp FROM ohlcv_data
                WHERE ticker = $1 AND interval = $2
                ORDER BY timestamp DESC LIMIT 1
                """,
                ticker, interval,
            )
        return row["timestamp"].isoformat() if row and row["timestamp"] else None

    def _cache_path(
        self,
        ticker: str,
        interval: str,
        seq_len: int,
        ts: str,
        buy_thresh: Optional[float] = None,
        sell_thresh: Optional[float] = None,
        lookahead: Optional[int] = None,
        barrier_atr_mult: Optional[float] = None,
    ) -> str:
        import hashlib
        # Include label-affecting config in the key so changing barriers or
        # lookahead invalidates stale cached labels automatically. Per-run
        # overrides take precedence over settings so equities and FX runs
        # never collide in the cache. `barrier_atr_mult` is part of the key
        # too — vol-normalised labels produce a completely different label
        # distribution than fixed thresholds, so they must not share a cache.
        bt = buy_thresh  if buy_thresh  is not None else settings.BUY_THRESHOLD
        st = sell_thresh if sell_thresh is not None else settings.SELL_THRESHOLD
        la = lookahead   if lookahead   is not None else settings.LOOKAHEAD_WINDOW
        am = barrier_atr_mult if barrier_atr_mult is not None else "none"
        # `v` bumps whenever the FEATURE_COLS / FEATURE_COLS_FX layout changes
        # so old-shape cached arrays don't get loaded into a wider-shape model.
        # v2 = +DXY/VIX/yield_spread/us_2y, vol features (57 FX cols)
        # v3 = +4h MTF features (61 FX cols) — regressed, rolled back
        # v4 = back to 57 FX cols (MTF disabled)
        key = f"{ticker}|{interval}|{seq_len}|{ts}|bt{bt}|st{st}|la{la}|am{am}|v4"
        digest = hashlib.md5(key.encode()).hexdigest()[:12]
        return os.path.join(_DATASET_CACHE_DIR, f"{ticker}_{interval}_{seq_len}_{digest}.npz")

    async def _load_ticker_cache(
        self, ticker: str, interval: str, seq_len: int,
        buy_thresh: Optional[float] = None,
        sell_thresh: Optional[float] = None,
        lookahead: Optional[int] = None,
        barrier_atr_mult: Optional[float] = None,
    ) -> Optional[tuple]:
        ts = await self._latest_bar_ts(ticker, interval)
        if ts is None:
            return None
        path = self._cache_path(
            ticker, interval, seq_len, ts,
            buy_thresh=buy_thresh, sell_thresh=sell_thresh,
            lookahead=lookahead, barrier_atr_mult=barrier_atr_mult,
        )
        if not os.path.exists(path):
            return None
        try:
            data = np.load(path, allow_pickle=False)
            return (data["p"], data["s"], data["c"], data["r"], data["t"])
        except Exception as e:
            logger.warning(f"⚠️  Cache read failed for {ticker}: {e}")
            return None

    async def _save_ticker_cache(
        self, ticker: str, interval: str, seq_len: int, arrays: tuple,
        buy_thresh: Optional[float] = None,
        sell_thresh: Optional[float] = None,
        lookahead: Optional[int] = None,
        barrier_atr_mult: Optional[float] = None,
    ) -> None:
        ts = await self._latest_bar_ts(ticker, interval)
        if ts is None:
            return
        os.makedirs(_DATASET_CACHE_DIR, exist_ok=True)
        path = self._cache_path(
            ticker, interval, seq_len, ts,
            buy_thresh=buy_thresh, sell_thresh=sell_thresh,
            lookahead=lookahead, barrier_atr_mult=barrier_atr_mult,
        )
        p, s, c, r, t = arrays
        try:
            np.savez(path, p=p, s=s, c=c, r=r, t=t)
        except Exception as e:
            logger.warning(f"⚠️  Cache write failed for {ticker}: {e}")

    async def train(
        self,
        tickers:    List[str],
        interval:   str = "1d",
        seq_len:    int = SEQUENCE_LEN,
        epochs:     int = 50,
        batch_size: int = 64,            # was 32 — better GPU utilisation on RTX 3050
        lr:         float = 3e-4,
        use_class_weights: bool = True,
        diagnostics: bool = False,       # when True, also runs WF + permutation importance
        asset_class: str = "equities",   # "equities" or "fx" — controls checkpoint folder
        buy_threshold: Optional[float] = None,    # override settings.BUY_THRESHOLD per-run
        sell_threshold: Optional[float] = None,   # override settings.SELL_THRESHOLD per-run
        lookahead_window: Optional[int] = None,   # override settings.LOOKAHEAD_WINDOW per-run
        barrier_atr_mult: Optional[float] = None, # when set: vol-normalised barriers (mult × ATR / close)
    ) -> dict:
        from app.ml.models.registry import _paths_for, _normalize_asset_class
        from app.ml.data.dataset import FEATURE_COLS_FX

        # Normalise asset class to "equities" or "fx"; raises for unknown values
        ac = _normalize_asset_class(asset_class)
        ac_paths = _paths_for(ac)
        logger.info(f"🏷️  Training asset_class={ac} (checkpoint dir: {ac_paths['dir']})")

        # Resolve effective label-thresholds. None → fall through to settings.
        eff_buy_t  = buy_threshold    if buy_threshold    is not None else None
        eff_sell_t = sell_threshold   if sell_threshold   is not None else None
        eff_la     = lookahead_window if lookahead_window is not None else None
        eff_atr_m  = barrier_atr_mult if barrier_atr_mult is not None else None
        if any(v is not None for v in (eff_buy_t, eff_sell_t, eff_la, eff_atr_m)):
            logger.info(
                f"🎚️  Label overrides: buy_thresh={eff_buy_t}, sell_thresh={eff_sell_t}, "
                f"lookahead={eff_la}, barrier_atr_mult={eff_atr_m}  (None = use settings default)"
            )

        logger.info(f"📦 Building dataset for {len(tickers)} tickers...")

        # ── Data quality pre-check ────────────────────────────────────────────
        tickers = await self._quality_filter(tickers, interval)
        if len(tickers) < 2:
            raise ValueError("Fewer than 2 tickers passed quality checks — ingest more data.")

        builder = DatasetBuilder(self.pool)

        # ── Per-ticker build for ticker-level split ───────────────────────────
        # Scaler is fitted on the train slice only (first 70% of each ticker)
        # so val/test statistics never leak into normalisation — the previous
        # full-series fit hid the real generalisation gap.
        #
        # Cache layer: per-ticker arrays are pickled to disk keyed by the
        # ticker's latest bar timestamp. If no new bars have arrived since
        # the last build, we reuse the cached arrays (saves ~1-3 min on
        # retrain-same-tickers runs).
        TRAIN_FRACTION = 0.70
        ticker_data: list = []
        scaler_params: dict = {}
        cache_hits = 0
        cache_misses = 0
        for ticker in tickers:
            try:
                cached = await self._load_ticker_cache(
                    ticker, interval, seq_len,
                    buy_thresh=eff_buy_t, sell_thresh=eff_sell_t,
                    lookahead=eff_la, barrier_atr_mult=eff_atr_m,
                )
                if cached is not None:
                    p, s, c, r, t = cached
                    cache_hits += 1
                else:
                    p, s, c, r, t = await builder._build_ticker(
                        ticker, interval, seq_len,
                        buy_thresh=eff_buy_t, sell_thresh=eff_sell_t,
                        lookahead=eff_la, barrier_atr_mult=eff_atr_m,
                    )
                    if p is not None and len(p) > 0:
                        await self._save_ticker_cache(
                            ticker, interval, seq_len, (p, s, c, r, t),
                            buy_thresh=eff_buy_t, sell_thresh=eff_sell_t,
                            lookahead=eff_la, barrier_atr_mult=eff_atr_m,
                        )
                    cache_misses += 1

                if p is not None and len(p) > 0:
                    train_end = max(1, int(len(p) * TRAIN_FRACTION))
                    p_norm, t_scaler = builder._normalise(p, fit_slice=(0, train_end))
                    scaler_params[ticker] = t_scaler
                    ticker_data.append((p_norm, s, c, r, t))
                    logger.info(f"✅ {ticker}: {len(p)} sequences")
            except Exception as e:
                logger.error(f"❌ {ticker}: {e}")
        logger.info(f"💾 Dataset cache: {cache_hits} hits, {cache_misses} misses")

        if not ticker_data:
            raise ValueError("No data built — run ingest + sentiment fetch first.")

        # Ticker-level chronological split — no leakage between tickers
        train_ds, val_ds, test_ds = split_dataset_per_ticker(ticker_data)

        # Flat arrays for class weights (from training portion only)
        all_cls = np.concatenate([d[2] for d in ticker_data])
        n_train_approx = int(len(all_cls) * 0.70)
        logger.info(
            f"📊 Split: train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}"
        )

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  drop_last=True)
        val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, drop_last=False)
        test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, drop_last=False)

        # FX uses 54 features (51 base + 3 macro globals). Equities stays at 51.
        n_feat = len(FEATURE_COLS_FX) if ac == "fx" else len(FEATURE_COLS)
        model_config = {
            "n_features":         n_feat,
            "seq_len":            seq_len,
            "d_model":            96,    # 64→96: more capacity for higher test accuracy
            "n_heads":            4,     # 96/4 = 24 dim/head
            "n_layers":           3,
            "d_ff":               384,   # 4× d_model
            "sent_input":         4,
            "sent_dim":           24,
            "mlp_hidden":         192,
            "dropout":            0.35,
            "use_cross_attention": True,
        }
        model = TradingFusionModel(**model_config)
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"🧠 Model parameters: {total_params:,}")

        # ── Class weights: fix HOLD dominance ────────────────────────────────
        # Made optional — over-aggressive weighting can collapse BUY recall by
        # pushing the model into a small, low-confidence region of the output
        # simplex. Disable to let natural class imbalance through.
        train_labels = all_cls[:n_train_approx]
        if use_class_weights:
            class_weights = compute_class_weights(train_labels, n_classes=3)
        else:
            class_weights = None
            logger.info("⚖️  Class weights disabled (use_class_weights=False)")

        # ── Versioned checkpoint name ─────────────────────────────────────────
        ckpt_name = new_checkpoint_name()

        trainer = Trainer(
            model,
            lr=lr,
            class_weights=class_weights,
            checkpoint_name=ckpt_name,
            checkpoint_dir=ac_paths["dir"],   # per-asset-class folder
        )
        train_results = trainer.fit(train_loader, val_loader, epochs=epochs)
        trainer.load_best()

        # ── Temperature calibration ───────────────────────────────────────────
        from app.ml.evaluation.calibration import TemperatureScaler
        cal = TemperatureScaler()
        temperature = cal.fit(model, val_loader)
        model_config["temperature"] = round(temperature, 6)
        logger.info(f"🌡️  Calibrated temperature T={temperature:.4f} saved to model config")

        # ── Threshold auto-tuning ─────────────────────────────────────────────
        tuned = self._tune_thresholds(model, val_loader, temperature, asset_class=ac)
        model_config["confidence_threshold"] = tuned["confidence_threshold"]
        model_config["margin_threshold"]     = tuned["margin_threshold"]
        logger.info(
            f"🎯 Tuned thresholds: conf={tuned['confidence_threshold']} "
            f"margin={tuned['margin_threshold']}"
        )

        # Permutation feature importance is the slowest part of evaluation
        # (~3-5 min). Skipped in fast mode; opt-in via diagnostics=True.
        evaluator    = ModelEvaluator(model, interval=interval)
        eval_results = evaluator.evaluate(test_loader, with_permutation=diagnostics)
        eval_report_path = os.path.join(ac_paths["dir"], "eval_report.json")
        evaluator.save_report(eval_results, path=eval_report_path)

        save_model_config(model_config,   asset_class=ac)
        save_scaler_params(scaler_params, asset_class=ac)  # dict: ticker → {mean, std}

        from app.services.storage_service import upload
        upload(f"{ac}/eval_report.json", eval_report_path)

        # ── Walk-forward summary (3 folds, same data) ─────────────────────────
        # Skipped in fast mode (~15-20 min saved). Pass diagnostics=True to
        # get the cross-regime accuracy estimates.
        if diagnostics:
            wf_summary = await self._embedded_walk_forward(
                ticker_data=ticker_data,
                model_config=model_config,
                epochs=min(epochs, 15),      # was 30 — folds plateau before 15
                batch_size=batch_size,
                lr=lr,
                n_splits=3,
                interval=interval,
            )
        else:
            wf_summary = {"skipped": True, "reason": "diagnostics=False (use ?diagnostics=true for WF)"}

        # ── Register version (promotes to best_model.pt if best val_loss) ────
        version_entry = register_version(
            checkpoint_name=ckpt_name,
            val_loss=train_results["best_val_loss"],
            val_acc=train_results["val_history"][-1]["acc"]
                    if train_results["val_history"] else 0.0,
            eval_metrics=eval_results,
            tickers=tickers,
            asset_class=ac,
        )

        # Reload singleton for this asset class so the API immediately serves the new model
        reload_model(ac)

        total_cls = all_cls
        return {
            "status": "success",
            "version": version_entry,
            "n_tickers": len(ticker_data),
            "n_train": len(train_ds),
            "n_val": len(val_ds),
            "started_at": datetime.now(timezone.utc).isoformat(),
            "dataset": {
                "total_samples": int(len(total_cls)),
                "train": len(train_ds),
                "val":   len(val_ds),
                "test":  len(test_ds),
                "class_dist": {
                    "hold": int((total_cls == 0).sum()),
                    "buy":  int((total_cls == 1).sum()),
                    "sell": int((total_cls == 2).sum()),
                },
                "class_weights": class_weights.tolist() if class_weights is not None else None,
            },
            "training":         train_results,
            "evaluation":       eval_results,
            "walk_forward":     wf_summary,
            "model_params": total_params,
            "checkpoint":   train_results["checkpoint"],
        }

    # ── Embedded walk-forward (called automatically inside train()) ──────────

    async def _embedded_walk_forward(
        self,
        ticker_data: list,
        model_config: dict,
        epochs: int,
        batch_size: int,
        lr: float,
        n_splits: int = 3,
        interval: str = "1d",
    ) -> dict:
        """
        Runs N expanding-window folds on the already-built ticker_data.
        Each fold trains a fresh model and evaluates on the held-out window.
        Returns an aggregate summary dict (not saved as a new checkpoint).
        """
        from app.ml.data.torch_dataset import TradingDataset, split_dataset_per_ticker

        # Flatten all ticker data for global walk-forward splits
        X_price_all  = np.concatenate([d[0] for d in ticker_data], axis=0)
        X_sent_all   = np.concatenate([d[1] for d in ticker_data], axis=0)
        y_cls_all    = np.concatenate([d[2] for d in ticker_data], axis=0)
        y_reg_all    = np.concatenate([d[3] for d in ticker_data], axis=0)
        y_timing_all = np.concatenate([d[4] for d in ticker_data], axis=0)

        N = len(y_cls_all)
        fold_size = N // (n_splits + 1)

        if fold_size < 30:
            logger.warning("⚠️  Embedded WF: not enough samples for walk-forward — skipping")
            return {"skipped": True, "reason": "insufficient data"}

        fold_results = []

        for fold in range(n_splits):
            train_end = int(N * (fold + 1) / (n_splits + 1))
            val_end   = int(N * (fold + 2) / (n_splits + 1))
            if train_end < 20 or (val_end - train_end) < 10:
                continue

            train_ds = TradingDataset(
                X_price_all[:train_end],        X_sent_all[:train_end],
                y_cls_all[:train_end],          y_reg_all[:train_end],
                y_timing_all[:train_end],
            )
            val_ds = TradingDataset(
                X_price_all[train_end:val_end], X_sent_all[train_end:val_end],
                y_cls_all[train_end:val_end],   y_reg_all[train_end:val_end],
                y_timing_all[train_end:val_end],
            )

            fold_loader_tr = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  drop_last=True)
            fold_loader_va = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, drop_last=False)

            fold_cw = compute_class_weights(y_cls_all[:train_end], n_classes=3)
            _INFERENCE_KEYS = {"temperature", "confidence_threshold", "margin_threshold"}
            arch_cfg = {k: v for k, v in model_config.items() if k not in _INFERENCE_KEYS}
            fold_model = TradingFusionModel(**arch_cfg)
            fold_trainer = Trainer(
                fold_model, lr=lr, class_weights=fold_cw,
                checkpoint_name=f"_wf_fold{fold+1}_tmp.pt",
            )
            fold_trainer.fit(fold_loader_tr, fold_loader_va, epochs=epochs)
            fold_trainer.load_best()

            fold_eval = ModelEvaluator(fold_model, interval=interval)
            metrics   = fold_eval.evaluate(fold_loader_va)

            fold_results.append({
                "fold":         fold + 1,
                "train_size":   train_end,
                "val_size":     val_end - train_end,
                "accuracy":     metrics["accuracy"],
                "f1_weighted":  metrics["f1_weighted"],
                "sharpe":       metrics["trading"]["sharpe_ratio"],
                "win_rate":     metrics["trading"]["win_rate"],
                "max_drawdown": metrics["trading"]["max_drawdown"],
                "class_recall": metrics["class_recall"],
            })
            logger.info(
                f"  WF fold {fold+1}/{n_splits}: "
                f"acc={metrics['accuracy']:.4f}  "
                f"sharpe={metrics['trading']['sharpe_ratio']:.4f}"
            )

            # Clean up temp checkpoint
            tmp = f"checkpoints/_wf_fold{fold+1}_tmp.pt"
            if os.path.exists(tmp):
                os.remove(tmp)

        if not fold_results:
            return {"skipped": True, "reason": "all folds had insufficient data"}

        def _avg(key):
            vals = [f[key] for f in fold_results if key in f]
            return round(float(np.mean(vals)), 4) if vals else 0.0

        summary = {
            "n_folds":      len(fold_results),
            "avg_accuracy": _avg("accuracy"),
            "std_accuracy": round(float(np.std([f["accuracy"] for f in fold_results])), 4),
            "avg_f1":       _avg("f1_weighted"),
            "avg_sharpe":   _avg("sharpe"),
            "std_sharpe":   round(float(np.std([f["sharpe"] for f in fold_results])), 4),
            "avg_win_rate": _avg("win_rate"),
            "avg_max_dd":   _avg("max_drawdown"),
            "folds":        fold_results,
        }

        # Save for the /ml/walkforward/result endpoint too
        import json as _json
        os.makedirs("checkpoints", exist_ok=True)
        with open("checkpoints/walkforward_result.json", "w") as f:
            _json.dump({"status": "success", "summary": summary, "folds": fold_results}, f, indent=2, default=str)

        logger.info(
            f"✅ Walk-forward: avg_acc={summary['avg_accuracy']:.4f} ± {summary['std_accuracy']:.4f}  "
            f"avg_sharpe={summary['avg_sharpe']:.4f} ± {summary['std_sharpe']:.4f}"
        )
        return summary

    # ── Walk-forward validation (standalone API endpoint) ─────────────────────

    async def walk_forward_validate(
        self,
        tickers:        List[str],
        interval:       str = "1d",
        seq_len:        int = SEQUENCE_LEN,
        n_splits:       int = 5,
        epochs:         int = 30,
        batch_size:     int = 32,
        lr:             float = 1e-3,
        min_train_ratio: float = 0.5,
    ) -> dict:
        """
        Walk-forward (expanding-window) validation.

        Splits the full chronological dataset into n_splits folds.
        For fold k, trains on the first (k+1)/n_splits of the data
        and evaluates on the next 1/n_splits slice.

        Returns per-fold and aggregate metrics so you can see how the
        model holds up across different market regimes.
        """
        logger.info(
            f"🔄 Walk-forward validation: {n_splits} folds, {len(tickers)} tickers"
        )
        builder = DatasetBuilder(self.pool)
        X_price, X_sent, y_cls, y_reg, y_timing, _ = await builder.build(
            tickers, interval, seq_len
        )

        N = len(y_cls)
        fold_size = N // (n_splits + 1)

        if fold_size < 50:
            raise ValueError(
                f"Not enough data for {n_splits} folds "
                f"(only {N} sequences). Ingest more data or reduce n_splits."
            )

        fold_results = []
        model_config = {
            "n_features": len(FEATURE_COLS),
            "seq_len":    seq_len,
            "d_model":    64,
            "n_heads":    4,
            "n_layers":   2,
            "d_ff":       256,
            "sent_input": 4,
            "sent_dim":   16,
            "mlp_hidden": 128,
            "dropout":    0.3,
        }

        for fold in range(n_splits):
            # Expanding train window: grows with each fold
            train_end = int(N * (fold + 1) / (n_splits + 1))
            val_end   = int(N * (fold + 2) / (n_splits + 1))

            if train_end < int(N * min_train_ratio):
                logger.info(f"⏩ Fold {fold+1}: skipping (insufficient train size)")
                continue

            logger.info(
                f"📂 Fold {fold+1}/{n_splits}: "
                f"train [0:{train_end}]  val [{train_end}:{val_end}]"
            )

            from app.ml.data.torch_dataset import TradingDataset

            # Per-fold normalisation: fit on this fold's train slice only.
            # build() now returns raw arrays so we control the fit scope here.
            fold_train_flat = X_price[:train_end].reshape(-1, X_price.shape[-1])
            fold_mean = fold_train_flat.mean(axis=0)
            fold_std  = np.maximum(fold_train_flat.std(axis=0), 1e-2)
            X_price_norm = np.clip((X_price - fold_mean) / fold_std, -10.0, 10.0).astype(np.float32)

            train_ds = TradingDataset(
                X_price_norm[:train_end],   X_sent[:train_end],
                y_cls[:train_end],          y_reg[:train_end],
                y_timing[:train_end],
            )
            val_ds = TradingDataset(
                X_price_norm[train_end:val_end], X_sent[train_end:val_end],
                y_cls[train_end:val_end],        y_reg[train_end:val_end],
                y_timing[train_end:val_end],
            )

            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  drop_last=True)
            val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, drop_last=False)

            train_labels  = y_cls[:train_end]
            class_weights = compute_class_weights(train_labels, n_classes=3)

            model   = TradingFusionModel(**model_config)
            trainer = Trainer(
                model,
                lr=lr,
                class_weights=class_weights,
                checkpoint_name=f"wf_fold{fold+1}_{new_checkpoint_name()}",
            )
            trainer.fit(train_loader, val_loader, epochs=epochs)
            trainer.load_best()

            evaluator = ModelEvaluator(model, interval=interval)
            metrics   = evaluator.evaluate(val_loader)

            fold_results.append({
                "fold":        fold + 1,
                "train_size":  train_end,
                "val_size":    val_end - train_end,
                "accuracy":    metrics["accuracy"],
                "f1_weighted": metrics["f1_weighted"],
                "sharpe":      metrics["trading"]["sharpe_ratio"],
                "win_rate":    metrics["trading"]["win_rate"],
                "max_drawdown":metrics["trading"]["max_drawdown"],
                "class_recall":metrics["class_recall"],
            })
            logger.info(
                f"  Fold {fold+1} → acc={metrics['accuracy']:.4f}  "
                f"sharpe={metrics['trading']['sharpe_ratio']:.4f}  "
                f"win={metrics['trading']['win_rate']:.4f}"
            )

        if not fold_results:
            raise ValueError("No folds completed — dataset too small.")

        # Aggregate summary
        def avg(key):
            vals = [f[key] for f in fold_results if key in f]
            return round(float(np.mean(vals)), 4) if vals else 0.0

        summary = {
            "n_folds":      len(fold_results),
            "avg_accuracy": avg("accuracy"),
            "avg_f1":       avg("f1_weighted"),
            "avg_sharpe":   avg("sharpe"),
            "avg_win_rate": avg("win_rate"),
            "avg_max_dd":   avg("max_drawdown"),
            "std_sharpe":   round(
                float(np.std([f["sharpe"] for f in fold_results])), 4
            ),
        }

        logger.info(
            f"✅ Walk-forward done — "
            f"avg_acc={summary['avg_accuracy']:.4f}  "
            f"avg_sharpe={summary['avg_sharpe']:.4f} ± {summary['std_sharpe']:.4f}"
        )

        return {
            "status":    "success",
            "summary":   summary,
            "folds":     fold_results,
            "n_tickers": len(tickers),
            "n_samples": N,
        }

    # ── Threshold auto-tuning ─────────────────────────────────────────────────

    def _tune_thresholds(
        self,
        model,
        val_loader,
        temperature: float = 1.0,
        asset_class: str = "equities",
    ) -> dict:
        """Sweep confidence and margin thresholds on the validation set.

        Progressive gate strategy:
          1. Look for combos with `precision ≥ strict_floor` AND `recall ≥ 1%`.
          2. If none qualify, relax to `precision ≥ relaxed_floor` AND `recall ≥ 0.5%`.
          3. If still none qualify, return the combo with the highest F1 found
             anywhere in the sweep — better than hard-coded defaults that
             cause the live system to fire on every borderline prediction.

        FX uses a lower precision floor (0.50/0.45) than equities (0.55/0.50)
        because vol-normalised FX labels naturally produce a more balanced but
        harder-to-discriminate classification task.
        """
        device = next(model.parameters()).device
        model.eval()

        all_probs  = []
        all_labels = []

        with torch.no_grad():
            for batch in val_loader:
                x_price, x_sent, y_cls = batch[0], batch[1], batch[2]
                dl, _, _ = model(x_price.to(device), x_sent.to(device))
                probs = torch.softmax(dl / max(temperature, 0.1), dim=-1).cpu().numpy()
                all_probs.append(probs)
                all_labels.append(y_cls.numpy())

        probs  = np.concatenate(all_probs)   # (N, 3)
        labels = np.concatenate(all_labels)  # (N,)

        max_probs  = probs.max(axis=1)
        pred_cls   = probs.argmax(axis=1)
        sorted_p   = np.sort(probs, axis=1)[:, ::-1]
        margins    = sorted_p[:, 0] - sorted_p[:, 1]
        is_signal  = (labels == 1) | (labels == 2)     # true BUY or SELL
        total_true = max(is_signal.sum(), 1)

        # Per-asset-class precision floors.
        if asset_class == "fx":
            strict_floor,  strict_recall  = 0.50, 0.01
            relaxed_floor, relaxed_recall = 0.45, 0.005
        else:
            strict_floor,  strict_recall  = 0.55, 0.01
            relaxed_floor, relaxed_recall = 0.50, 0.005

        # Sweep every combo once; record best-F1 along three independent tracks
        # so we can fall back from strict → relaxed → any combo.
        strict_best  = (-1.0, None)   # (f1, combo) for precision ≥ strict_floor
        relaxed_best = (-1.0, None)   # for precision ≥ relaxed_floor
        any_best     = (-1.0, None)   # any combo with at least one prediction
        any_precision = 0.0
        any_recall    = 0.0

        for conf_t in np.arange(0.50, 0.85, 0.02):
            for margin_t in np.arange(0.05, 0.35, 0.02):
                passes  = (max_probs >= conf_t) & (margins >= margin_t) & (pred_cls != 0)
                n_pred  = passes.sum()
                if n_pred < 1:
                    continue

                tp        = ((pred_cls == labels) & passes).sum()
                precision = tp / n_pred
                recall    = tp / total_true
                f1        = 2 * precision * recall / max(precision + recall, 1e-9)

                combo = {
                    "confidence_threshold": round(float(conf_t), 3),
                    "margin_threshold":     round(float(margin_t), 3),
                }

                if f1 > any_best[0]:
                    any_best       = (f1, combo)
                    any_precision  = precision
                    any_recall     = recall
                if precision >= relaxed_floor and recall >= relaxed_recall and f1 > relaxed_best[0]:
                    relaxed_best = (f1, combo)
                if precision >= strict_floor and recall >= strict_recall and f1 > strict_best[0]:
                    strict_best = (f1, combo)

        # Pick the best result available along the strict→relaxed→any ladder
        if strict_best[1] is not None:
            tier = "strict"
            best_f1, best = strict_best
        elif relaxed_best[1] is not None:
            tier = "relaxed"
            best_f1, best = relaxed_best
        elif any_best[1] is not None:
            tier = "any (no precision gate)"
            best_f1, best = any_best
        else:
            tier = "default"
            best_f1, best = -1.0, {"confidence_threshold": 0.55, "margin_threshold": 0.10}

        logger.info(
            f"🎯 Threshold sweep [{asset_class}/{tier}]: "
            f"conf={best['confidence_threshold']}  margin={best['margin_threshold']}  "
            f"F1={best_f1:.4f}  any-best precision={any_precision:.3f} recall={any_recall:.3f}"
        )
        return best

    # ── Data quality pre-check ────────────────────────────────────────────────

    async def _quality_filter(self, tickers: List[str], interval: str) -> List[str]:
        """
        Remove tickers with too many missing bars, no indicator data, or a
        bar-count that is much lower than the median across the cohort
        (catches stale/renamed tickers like FB whose data ends in 2022 and
        contaminates training with a different distribution).
        """
        # Pass 1 — collect bar counts and per-ticker missing-close ratios
        stats: list[tuple[str, int, float]] = []
        async with self.pool.acquire() as conn:
            for ticker in tickers:
                row = await conn.fetchrow("""
                    SELECT
                        COUNT(*)                                              AS total,
                        COUNT(*) FILTER (WHERE close IS NULL)                 AS null_close
                    FROM ohlcv_data
                    WHERE ticker = $1 AND interval = $2
                """, ticker, interval)
                total = int(row["total"]) if row else 0
                null_pct = (int(row["null_close"]) / total) if total else 1.0
                stats.append((ticker, total, null_pct))

        if not stats:
            return []

        bar_counts = [t for _, t, _ in stats if t > 0]
        median_bars = int(np.median(bar_counts)) if bar_counts else 0
        # Drop tickers with <50% of the cohort's median bars — guards against
        # delisted symbols (FB→META rename) being trained alongside live ones.
        min_bars_floor = max(60, int(median_bars * 0.5))

        good: list[str] = []
        for ticker, total, null_pct in stats:
            if total < 60:
                logger.warning(f"⚠️  Quality filter: {ticker} has <60 bars — skipped")
                continue
            if total < min_bars_floor:
                logger.warning(
                    f"⚠️  Quality filter: {ticker} has {total} bars "
                    f"(< 50% of cohort median {median_bars}) — skipped"
                )
                continue
            if null_pct > 0.05:
                logger.warning(
                    f"⚠️  Quality filter: {ticker} has {null_pct:.1%} null closes — skipped"
                )
                continue
            good.append(ticker)

        logger.info(
            f"✅ Quality filter: {len(good)}/{len(tickers)} tickers passed "
            f"(median bars={median_bars}, floor={min_bars_floor})"
        )
        return good

    # ── Single-ticker inference ───────────────────────────────────────────────

    async def predict_ticker(self, ticker: str, interval: str = "1d") -> dict:
        import json as _json
        from app.ml.models.registry import get_model, load_scaler_params, _paths_for
        from app.services.cache_service import get_cache
        from app.core.asset_class import asset_class_for
        from app.core.config import settings

        # Route to the right model checkpoint based on the ticker's asset class.
        # `asset_class_for` returns 'equity' / 'fx_major' / 'fx_metal'; the
        # registry normalises those to 'equities' / 'fx' internally.
        ac = asset_class_for(ticker)
        paths = _paths_for(ac)
        config_path = paths["config"]

        _mcfg: dict = {}
        try:
            if os.path.exists(config_path):
                with open(config_path) as _f:
                    _mcfg = _json.load(_f)
        except Exception:
            pass

        # Cache TTL scales inversely with bar duration: a 1d prediction stays
        # valid much longer than a 5m one because a new bar arrives less often.
        ttl_by_interval = {"1d": 300, "1h": 600, "30m": 300, "15m": 180, "5m": 120}
        ttl = ttl_by_interval.get(interval, 300)

        cache = await get_cache()
        cache_key = f"predict:{ticker.upper()}:{interval}"
        cached = await cache.get(cache_key)
        if cached is not None:
            logger.info(f"⚡ Cache hit: {cache_key}")
            return cached

        model = get_model(ac)
        if model is None:
            return {"error": f"{paths['asset_class']} model not trained yet. Run POST /api/v1/ml/train first."}

        scaler = load_scaler_params(ac)

        builder = DatasetBuilder(self.pool)
        X_price, X_sent, _, _, _, _ = await builder.build(
            [ticker], interval, sequence_len=SEQUENCE_LEN
        )

        if X_price is None or len(X_price) == 0:
            return {"error": f"Not enough data for {ticker}"}

        # Read the raw last close *before* normalisation — required for
        # converting predicted % targets back to dollars. The previous
        # implementation read X_price[-1,-1,3] *after* z-score normalisation,
        # producing nonsense dollar values in the UI.
        async with self.pool.acquire() as conn:
            close_row = await conn.fetchrow(
                """
                SELECT close FROM ohlcv_data
                WHERE ticker = $1 AND interval = $2
                ORDER BY timestamp DESC LIMIT 1
                """,
                ticker.upper(), interval,
            )
            # Latest indicator snapshot — used for event-window filtering and
            # ATR-based SL/TP scaling at execution time.
            ind_row = await conn.fetchrow(
                """
                SELECT atr_14, fomc_days, cpi_days, nfp_days, earnings_days
                FROM indicators
                WHERE ticker = $1 AND interval = $2
                ORDER BY timestamp DESC LIMIT 1
                """,
                ticker.upper(), interval,
            )
        if not close_row or close_row["close"] is None:
            return {"error": f"No recent close price for {ticker}"}
        current_close = float(close_row["close"])

        atr_14         = float(ind_row["atr_14"])         if ind_row and ind_row["atr_14"]         is not None else None
        fomc_days      = float(ind_row["fomc_days"])      if ind_row and ind_row["fomc_days"]      is not None else 999.0
        cpi_days       = float(ind_row["cpi_days"])       if ind_row and ind_row["cpi_days"]       is not None else 999.0
        nfp_days       = float(ind_row["nfp_days"])       if ind_row and ind_row["nfp_days"]       is not None else 999.0
        earnings_days  = float(ind_row["earnings_days"])  if ind_row and ind_row["earnings_days"]  is not None else 999.0

        if scaler:
            # Per-ticker scaler: dict keyed by ticker
            ticker_scaler = scaler.get(ticker.upper()) or scaler.get(ticker)
            if ticker_scaler:
                mean    = np.array(ticker_scaler["mean"], dtype=np.float32)
                std     = np.maximum(np.array(ticker_scaler["std"], dtype=np.float32), 1e-2)
                X_price = np.clip((X_price - mean) / std, -10.0, 10.0)
            elif isinstance(scaler, dict) and "mean" in scaler:
                mean    = np.array(scaler["mean"], dtype=np.float32)
                std     = np.maximum(np.array(scaler["std"], dtype=np.float32), 1e-2)
                X_price = np.clip((X_price - mean) / std, -10.0, 10.0)

        # Mirror the trainer's NaN scrubbing: some indicators (e.g. MFI for FX
        # tickers where yfinance's volume series is constant zero) produce
        # NaNs through their entire history. Training zeroed them out via
        # `torch.nan_to_num` so the model learned "0 = no signal" — inference
        # must do the same or the softmax over NaN logits gives NaN probs.
        X_price = np.nan_to_num(X_price, nan=0.0, posinf=0.0, neginf=0.0)
        X_sent  = np.nan_to_num(X_sent,  nan=0.0, posinf=0.0, neginf=0.0)

        device  = next(model.parameters()).device
        x_price = torch.tensor(X_price[-1:], dtype=torch.float32).to(device)
        x_sent  = torch.tensor(X_sent[-1:],  dtype=torch.float32).to(device)

        # Temperature may be NaN if calibration failed (e.g., FX runs where
        # validation set has too few non-HOLD samples to fit a scaler) —
        # treat NaN as 1.0 (uncalibrated) instead of producing NaN softmax
        # probabilities that crash the whole filter chain.
        _temp_raw = _mcfg.get("temperature", 1.0)
        try:
            temperature = float(_temp_raw)
            if not np.isfinite(temperature) or temperature <= 0:
                temperature = 1.0
        except (TypeError, ValueError):
            temperature = 1.0
        _CONFIDENCE_THRESHOLD = float(_mcfg.get("confidence_threshold", 0.58))
        _MARGIN_THRESHOLD     = float(_mcfg.get("margin_threshold",     0.15))
        # Per-asset-class action suppression: model_config can blacklist
        # specific actions (e.g. FX disables SELL because per-class precision
        # was anti-edge in validation). The filter chain downgrades these to
        # HOLD before they reach the scoring stage.
        _DISABLED_ACTIONS: set = set(_mcfg.get("disabled_actions") or [])

        current_ts = datetime.now(timezone.utc)
        # 8 MC samples gives ~80% of the uncertainty signal at ~40% the latency
        # of n=20 — solid trade-off for live inference.
        result     = model.mc_predict(
            x_price, x_sent,
            n_samples=8,
            temperature=temperature,
            current_ts=current_ts,
            interval=interval,
        )

        action      = result["action"][0]
        confidence  = result["confidence"][0]
        uncertainty = float(result.get("uncertainty", [0.0])[0])
        prob_hold   = result["probabilities"]["hold"][0]
        prob_buy    = result["probabilities"]["buy"][0]
        prob_sell   = result["probabilities"]["sell"][0]
        probs_sorted = sorted([prob_hold, prob_buy, prob_sell], reverse=True)
        prob_margin  = probs_sorted[0] - probs_sorted[1]

        # The regression and timing heads still exist in the architecture for
        # checkpoint compatibility, but their outputs are ignored — SL/TP/
        # entry are computed mechanically below from ATR. The model is used
        # for direction only.
        entry_offset_pct = 0.0
        net_pct          = 0.0
        raw_bars         = 1.0
        timing_bucket    = 0
        timing_bars      = 1.0

        # ── ATR-mechanical SL/TP magnitudes ─────────────────────────────────────
        # Direction-only magnitudes — actual dollar levels are recomputed below
        # once we know which side of the entry the SL/TP should sit on. Using
        # action="BUY" here just gives us the symmetric |distance| values; we
        # branch into actual BUY/SELL/HOLD price math after the filter chain.
        from app.strategies.atr_levels import compute_atr_levels
        _mag = compute_atr_levels(atr_14, current_close, "BUY")
        abs_sl_pct = max(_mag["abs_sl_pct"], settings.MIN_PREDICTED_SL_PCT)
        abs_tp_pct = _mag["abs_tp_pct"]

        # ── Filter chain — each gate can downgrade action to HOLD ───────────────
        reject_reasons: list = []

        # Gate 0: disabled-action veto (per-asset-class class suppression)
        # FX disables SELL because validation showed it was anti-edge (35%
        # precision). The list comes from model_config.json:disabled_actions.
        if action in _DISABLED_ACTIONS:
            reject_reasons.append(f"action_{action.lower()}_disabled_for_{paths['asset_class']}")
            action = "HOLD"

        # Gate 1: confidence + margin (val-tuned thresholds)
        if action != "HOLD" and (confidence < _CONFIDENCE_THRESHOLD or prob_margin < _MARGIN_THRESHOLD):
            reject_reasons.append(f"conf={confidence:.3f}/margin={prob_margin:.3f}")
            action = "HOLD"

        # Gate 2: uncertainty (MC dropout std at predicted class)
        if action != "HOLD" and uncertainty > settings.UNCERTAINTY_MAX:
            reject_reasons.append(f"uncertainty={uncertainty:.3f}>{settings.UNCERTAINTY_MAX}")
            action = "HOLD"

        # Gate 3: R:R ratio — only trade when predicted reward justifies risk
        predicted_rr = abs_tp_pct / abs_sl_pct
        if action != "HOLD" and predicted_rr < settings.MIN_RR_RATIO:
            reject_reasons.append(f"RR={predicted_rr:.2f}<{settings.MIN_RR_RATIO}")
            action = "HOLD"

        # Gate 4: macro-event windows — gap risk dominates short-term edge
        next_macro = min(fomc_days, cpi_days, nfp_days)
        if action != "HOLD" and next_macro < settings.EVENT_DAYS_MIN:
            reject_reasons.append(f"macro_event_in_{next_macro:.0f}d")
            action = "HOLD"
        if action != "HOLD" and earnings_days < settings.EARNINGS_DAYS_MIN:
            reject_reasons.append(f"earnings_in_{earnings_days:.0f}d")
            action = "HOLD"

        # Gate 5: expected value — confidence × tp − (1-conf) × sl
        # The single most important filter: if EV ≤ 0, this is a losing trade
        # in expectation no matter how high confidence looks.
        expected_value = confidence * abs_tp_pct - (1.0 - confidence) * abs_sl_pct
        if action != "HOLD" and expected_value < settings.MIN_EXPECTED_VALUE:
            reject_reasons.append(f"EV={expected_value:.4f}")
            action = "HOLD"

        if reject_reasons:
            logger.info(f"🔇 {ticker}: filtered → HOLD ({', '.join(reject_reasons)})")

        # ── Position sizing: quarter-Kelly with edge/RR ─────────────────────────
        # Kelly fraction: f* = (p·b − q) / b   where
        #   p = win probability ≈ confidence
        #   q = 1 − p
        #   b = R:R ratio
        # Clamped to [0, 1]; multiplied by KELLY_FRACTION (0.25 = quarter-Kelly)
        # for safety against overestimated edge.
        if action != "HOLD":
            p = confidence
            b = predicted_rr
            kelly_full     = max(0.0, (p * b - (1.0 - p)) / b) if b > 0 else 0.0
            kelly_fraction = min(1.0, kelly_full * settings.KELLY_FRACTION)
        else:
            kelly_full     = 0.0
            kelly_fraction = 0.0

        # ── Bars-to-entry cap ──────────────────────────────────────────────────
        max_bars = float(getattr(settings, "MAX_BARS_TO_ENTRY", 10))
        bars_capped = raw_bars > max_bars
        bars_to_entry = min(max(raw_bars, 1.0), max_bars)
        entry_time = entry_time_from_bars(current_ts, bars_to_entry, interval)

        # ── Dollar levels — ATR-mechanical, no model regression involved ────────
        # entry is the current close (no model-predicted offset); SL and TP are
        # entry ± K × ATR. Both `*_dollar` and `atr_*_dollar` columns get the
        # same ATR-based value so downstream (signals table, execution layer,
        # dashboard) all see consistent realistic levels.
        entry_dollar = current_close
        if action == "BUY":
            sl_dollar = entry_dollar * (1 - abs_sl_pct)
            tp_dollar = entry_dollar * (1 + abs_tp_pct)
        elif action == "SELL":
            sl_dollar = entry_dollar * (1 + abs_sl_pct)
            tp_dollar = entry_dollar * (1 - abs_tp_pct)
        else:  # HOLD
            sl_dollar = entry_dollar
            tp_dollar = entry_dollar
        atr_sl_dollar = sl_dollar
        atr_tp_dollar = tp_dollar

        net_dollar = abs(tp_dollar - entry_dollar) - abs(sl_dollar - entry_dollar)

        output = {
            "ticker":    ticker.upper(),
            "interval":  interval,
            "action":      action,
            "confidence":  round(confidence, 4),
            "uncertainty": round(uncertainty, 4),
            "probabilities": {
                "hold": round(result["probabilities"]["hold"][0], 4),
                "buy":  round(result["probabilities"]["buy"][0],  4),
                "sell": round(result["probabilities"]["sell"][0], 4),
            },
            # Dollar levels derived from predicted % targets × anchor close
            # Now uses the model's PREDICTED tp_pct (was overridden with 2:1 R:R)
            "entry_price":  round(entry_dollar, 4),
            "stop_loss":    round(sl_dollar,    4),
            "take_profit":  round(tp_dollar,    4),
            "net_profit":   round(net_dollar,   4),
            "bars_to_entry": round(bars_to_entry, 2),
            "bars_to_entry_capped": bars_capped,
            "entry_time":   entry_time.isoformat(),
            "entry_time_label": _entry_label(entry_time, interval),
            # ── Profitability fields ───────────────────────────────────────────
            # Use these on the execution side to size positions and gate trades.
            "expected_value":   round(expected_value, 6),  # confidence·TP − (1−c)·SL
            "predicted_rr":     round(predicted_rr, 3),    # tp/sl ratio
            "kelly_full":       round(kelly_full, 4),      # raw Kelly fraction
            "kelly_fraction":   round(kelly_fraction, 4),  # quarter-Kelly position size (0..1)
            "risk_per_trade":   round(settings.MAX_RISK_PER_TRADE, 4),
            "reject_reasons":   reject_reasons,            # populated only when filtered to HOLD
            # ATR-based SL/TP alternative — execution layer can substitute these
            # for volatility-adaptive stops instead of the model's % distances.
            "atr_levels": {
                "stop_loss":   round(atr_sl_dollar, 4),
                "take_profit": round(atr_tp_dollar, 4),
                "atr_14":      round(atr_14, 6) if atr_14 else None,
            },
            # Event proximity flags — caller may want to surface these in the UI
            "event_proximity": {
                "fomc_days":     round(fomc_days, 1),
                "cpi_days":      round(cpi_days, 1),
                "nfp_days":      round(nfp_days, 1),
                "earnings_days": round(earnings_days, 1),
            },
            # ATR-mechanical levels (regression/timing heads no longer used)
            "predicted": {
                "entry_offset_pct": 0.0,
                "stop_loss_pct":    round(abs_sl_pct, 6),
                "take_profit_pct":  round(abs_tp_pct, 6),
                "net_profit_pct":   round(abs_tp_pct - abs_sl_pct, 6),
                "bars_to_entry":    1.0,
            },
            "timing": {
                "bucket":       0,
                "bucket_label": "bar+1",
                "bars":         1.0,
                "probs":        [1.0, 0.0, 0.0, 0.0],
            },
            "anchor_close": round(current_close, 4),
            "source":       "ml_fx" if ac in ("fx_major", "fx_metal", "fx") else "ml_equities",
            "generated_at": current_ts.isoformat(),
        }

        await cache.set(cache_key, output, ttl=ttl)
        return output


def _entry_label(entry_time: datetime, interval: str) -> str:
    if interval == "1d":
        return (
            f"Next trading day — "
            f"{entry_time.strftime('%A %b %d, %Y at %I:%M %p')} EST"
        )
    if interval == "1h":
        return f"Next candle — {entry_time.strftime('%b %d, %Y at %H:%M')} UTC"
    return entry_time.isoformat()

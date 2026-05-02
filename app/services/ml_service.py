import os
import torch
import numpy as np
import logging
import asyncpg
from torch.utils.data import DataLoader
from datetime import datetime, timezone
from typing import List

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

logger = logging.getLogger(__name__)


class MLService:
    def __init__(self, pool: asyncpg.Pool):
        self.pool = pool

    async def train(
        self,
        tickers:    List[str],
        interval:   str = "1d",
        seq_len:    int = SEQUENCE_LEN,
        epochs:     int = 50,
        batch_size: int = 32,
        lr:         float = 3e-4,
        use_class_weights: bool = True,
    ) -> dict:
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
        TRAIN_FRACTION = 0.70
        ticker_data: list = []
        scaler_params: dict = {}
        for ticker in tickers:
            try:
                p, s, c, r, t = await builder._build_ticker(ticker, interval, seq_len)
                if p is not None and len(p) > 0:
                    train_end = max(1, int(len(p) * TRAIN_FRACTION))
                    p_norm, t_scaler = builder._normalise(p, fit_slice=(0, train_end))
                    scaler_params[ticker] = t_scaler
                    ticker_data.append((p_norm, s, c, r, t))
                    logger.info(f"✅ {ticker}: {len(p)} sequences")
            except Exception as e:
                logger.error(f"❌ {ticker}: {e}")

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

        model_config = {
            "n_features": len(FEATURE_COLS),
            "seq_len":    seq_len,
            "d_model":    128,
            "n_heads":    8,
            "n_layers":   2,
            "d_ff":       256,
            "sent_input": 4,
            "sent_dim":   32,
            "mlp_hidden": 128,
            "dropout":    0.30,
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
            timing_weight=0.0,   # timing labels are noisy (argmin of 3 bars); disable to protect shared repr
            checkpoint_name=ckpt_name,
        )
        train_results = trainer.fit(train_loader, val_loader, epochs=epochs)
        trainer.load_best()

        evaluator    = ModelEvaluator(model, interval=interval)
        eval_results = evaluator.evaluate(test_loader)
        evaluator.save_report(eval_results)

        save_model_config(model_config)
        save_scaler_params(scaler_params)  # now a dict: ticker → {mean, std}

        # ── Walk-forward summary (3 folds, same data) ─────────────────────────
        # Runs after the final model is trained to give N accuracy estimates
        # across different market regimes rather than a single 70/15/15 split.
        wf_summary = await self._embedded_walk_forward(
            ticker_data=ticker_data,
            model_config=model_config,
            epochs=min(epochs, 30),
            batch_size=batch_size,
            lr=lr,
            n_splits=3,
            interval=interval,
        )

        # ── Register version (promotes to best_model.pt if best val_loss) ────
        version_entry = register_version(
            checkpoint_name=ckpt_name,
            val_loss=train_results["best_val_loss"],
            val_acc=train_results["val_history"][-1]["acc"]
                    if train_results["val_history"] else 0.0,
            eval_metrics=eval_results,
            tickers=tickers,
        )

        # Reload singleton so API immediately serves the new model
        reload_model()

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
            fold_model = TradingFusionModel(**model_config)
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

            train_ds = TradingDataset(
                X_price[:train_end],        X_sent[:train_end],
                y_cls[:train_end],          y_reg[:train_end],
                y_timing[:train_end],
            )
            val_ds = TradingDataset(
                X_price[train_end:val_end], X_sent[train_end:val_end],
                y_cls[train_end:val_end],   y_reg[train_end:val_end],
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
        from app.ml.models.registry import get_model, load_scaler_params
        from app.services.cache_service import get_cache
        from app.core.config import settings

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

        model = get_model()
        if model is None:
            return {"error": "Model not trained yet. Run POST /api/v1/ml/train first."}

        scaler = load_scaler_params()

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
            sma_row = await conn.fetchrow(
                """
                SELECT sma_20 FROM indicators
                WHERE ticker = $1 AND interval = $2
                ORDER BY timestamp DESC LIMIT 1
                """,
                ticker.upper(), interval,
            )
        if not close_row or close_row["close"] is None:
            return {"error": f"No recent close price for {ticker}"}
        current_close = float(close_row["close"])
        current_sma20 = float(sma_row["sma_20"]) if sma_row and sma_row["sma_20"] is not None else None

        if scaler:
            # Per-ticker scaler: dict keyed by ticker
            ticker_scaler = scaler.get(ticker.upper()) or scaler.get(ticker)
            if ticker_scaler:
                mean    = np.array(ticker_scaler["mean"], dtype=np.float32)
                std     = np.array(ticker_scaler["std"],  dtype=np.float32)
                X_price = (X_price - mean) / (std + 1e-8)
            elif isinstance(scaler, dict) and "mean" in scaler:
                # Backwards-compat: old global scaler format
                mean    = np.array(scaler["mean"], dtype=np.float32)
                std     = np.array(scaler["std"],  dtype=np.float32)
                X_price = (X_price - mean) / (std + 1e-8)

        device  = next(model.parameters()).device
        x_price = torch.tensor(X_price[-1:], dtype=torch.float32).to(device)
        x_sent  = torch.tensor(X_sent[-1:],  dtype=torch.float32).to(device)

        current_ts = datetime.now(timezone.utc)
        result     = model.predict(x_price, x_sent, current_ts=current_ts, interval=interval)

        action     = result["action"][0]
        confidence = result["confidence"][0]
        prob_hold  = result["probabilities"]["hold"][0]
        prob_buy   = result["probabilities"]["buy"][0]
        prob_sell  = result["probabilities"]["sell"][0]
        probs_sorted = sorted([prob_hold, prob_buy, prob_sell], reverse=True)
        prob_margin  = probs_sorted[0] - probs_sorted[1]

        # Filter 1: confidence + margin — both must pass.
        # A signal with BUY=0.55, SELL=0.30 is not actionable; the dominant
        # class must lead the second by ≥0.15 to indicate real separation.
        _CONFIDENCE_THRESHOLD = 0.58
        _MARGIN_THRESHOLD     = 0.15
        if action != "HOLD" and (confidence < _CONFIDENCE_THRESHOLD or prob_margin < _MARGIN_THRESHOLD):
            logger.info(
                f"🔇 {ticker}: {action} filtered — conf={confidence:.3f} margin={prob_margin:.3f} → HOLD"
            )
            action = "HOLD"

        # Filter 2: trend confirmation — BUY only above SMA20, SELL only below.
        # Counter-trend entries are the most common source of bad signals.
        if action == "BUY" and current_sma20 and current_close < current_sma20:
            logger.info(
                f"🔇 {ticker}: BUY rejected — price {current_close:.2f} below SMA20 {current_sma20:.2f}"
            )
            action = "HOLD"
        elif action == "SELL" and current_sma20 and current_close > current_sma20:
            logger.info(
                f"🔇 {ticker}: SELL rejected — price {current_close:.2f} above SMA20 {current_sma20:.2f}"
            )
            action = "HOLD"

        # ── All five regression outputs are genuine model predictions ─────────
        entry_offset_pct = float(result["entry_price"][0])   # predicted % from close
        sl_pct           = float(result["stop_loss"][0])
        tp_pct           = float(result["take_profit"][0])
        net_pct          = float(result["net_profit"][0])
        raw_bars         = float(result["bars_to_entry"][0])
        timing_bucket    = int(result["timing_bucket"][0])
        timing_bars      = float(result["timing_bars"][0])

        max_bars = float(getattr(settings, "MAX_BARS_TO_ENTRY", 10))
        bars_capped = raw_bars > max_bars
        bars_to_entry = min(max(raw_bars, 1.0), max_bars)
        entry_time = entry_time_from_bars(current_ts, bars_to_entry, interval)

        # Convert predicted % offsets to dollar levels
        entry_dollar = current_close * (1 + entry_offset_pct)
        if action == "BUY":
            sl_dollar   = entry_dollar * (1 + sl_pct)
            sl_distance = entry_dollar - sl_dollar          # always positive
            tp_dollar   = entry_dollar + sl_distance * 2.0  # fixed 2:1 R:R
        elif action == "SELL":
            sl_dollar   = entry_dollar * (1 - sl_pct)
            sl_distance = sl_dollar - entry_dollar          # always positive
            tp_dollar   = entry_dollar - sl_distance * 2.0  # fixed 2:1 R:R
        else:  # HOLD
            sl_dollar = entry_dollar
            tp_dollar = entry_dollar
        net_dollar = abs(tp_dollar - entry_dollar) - abs(sl_dollar - entry_dollar)

        output = {
            "ticker":    ticker.upper(),
            "interval":  interval,
            "action":    action,
            "confidence": round(confidence, 4),
            "probabilities": {
                "hold": round(result["probabilities"]["hold"][0], 4),
                "buy":  round(result["probabilities"]["buy"][0],  4),
                "sell": round(result["probabilities"]["sell"][0], 4),
            },
            # Dollar levels derived from predicted % targets × anchor close
            "entry_price":  round(entry_dollar, 4),
            "stop_loss":    round(sl_dollar,    4),
            "take_profit":  round(tp_dollar,    4),
            "net_profit":   round(net_dollar,   4),
            "bars_to_entry": round(bars_to_entry, 2),
            "bars_to_entry_capped": bars_capped,
            "entry_time":   entry_time.isoformat(),
            "entry_time_label": _entry_label(entry_time, interval),
            # All five raw model predictions (% / float)
            "predicted": {
                "entry_offset_pct": round(entry_offset_pct, 6),
                "stop_loss_pct":    round(sl_pct, 6),
                "take_profit_pct":  round(tp_pct, 6),
                "net_profit_pct":   round(net_pct, 6),
                "bars_to_entry":    round(raw_bars, 4),
            },
            # Timing head output
            "timing": {
                "bucket":       timing_bucket,
                "bucket_label": ["bar+1", "bar+2", "bar+3", "HOLD"][timing_bucket],
                "bars":         timing_bars,
                "probs":        [round(p, 4) for p in result["timing_probs"][0]],
            },
            "anchor_close": round(current_close, 4),
            "source":       "ml_model",
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

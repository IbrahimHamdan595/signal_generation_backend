import torch
import numpy as np
import logging
import asyncpg
from torch.utils.data import DataLoader
from datetime import datetime, timezone
from typing import List

from app.ml.data.dataset import DatasetBuilder, FEATURE_COLS, SEQUENCE_LEN
from app.ml.data.entry_time import entry_time_from_bars
from app.ml.data.torch_dataset import split_dataset, split_dataset_per_ticker
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
        lr:         float = 1e-3,
    ) -> dict:
        logger.info(f"📦 Building dataset for {len(tickers)} tickers...")

        # ── Data quality pre-check ────────────────────────────────────────────
        tickers = await self._quality_filter(tickers, interval)
        if len(tickers) < 2:
            raise ValueError("Fewer than 2 tickers passed quality checks — ingest more data.")

        builder = DatasetBuilder(self.pool)

        # ── Per-ticker build for ticker-level split ───────────────────────────
        ticker_data: list = []
        scaler_params: dict = {}
        for ticker in tickers:
            try:
                p, s, c, r = await builder._build_ticker(ticker, interval, seq_len)
                if p is not None and len(p) > 0:
                    p_norm, t_scaler = builder._normalise(p)
                    scaler_params[ticker] = t_scaler
                    ticker_data.append((p_norm, s, c, r))
                    logger.info(f"✅ {ticker}: {len(p)} sequences")
            except Exception as e:
                logger.error(f"❌ {ticker}: {e}")

        if not ticker_data:
            raise ValueError("No data built — run ingest + sentiment fetch first.")

        # Ticker-level chronological split — no leakage between tickers
        train_ds, val_ds, test_ds = split_dataset_per_ticker(ticker_data)

        # Reconstruct flat arrays for class weights (from training portion only)
        all_cls = np.concatenate([d[2] for d in ticker_data])
        n_train_approx = int(len(all_cls) * 0.70)
        logger.info(
            f"📊 Split: train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}"
        )

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
        test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

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
            "dropout":    0.1,
        }
        model = TradingFusionModel(**model_config)
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"🧠 Model parameters: {total_params:,}")

        # ── Class weights: fix HOLD dominance ────────────────────────────────
        train_labels  = all_cls[:n_train_approx]
        class_weights = compute_class_weights(train_labels, n_classes=3)

        # ── Versioned checkpoint name ─────────────────────────────────────────
        ckpt_name = new_checkpoint_name()

        trainer = Trainer(
            model,
            lr=lr,
            class_weights=class_weights,
            checkpoint_name=ckpt_name,
        )
        train_results = trainer.fit(train_loader, val_loader, epochs=epochs)
        trainer.load_best()

        evaluator    = ModelEvaluator(model)
        eval_results = evaluator.evaluate(test_loader)
        evaluator.save_report(eval_results)

        save_model_config(model_config)
        save_scaler_params(scaler_params)  # now a dict: ticker → {mean, std}

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
                "class_weights": class_weights.tolist(),
            },
            "training":    train_results,
            "evaluation":  eval_results,
            "model_params": total_params,
            "checkpoint":   train_results["checkpoint"],
        }

    # ── Walk-forward validation ───────────────────────────────────────────────

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
        X_price, X_sent, y_cls, y_reg, _ = await builder.build(
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
            "dropout":    0.1,
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
                X_price[:train_end], X_sent[:train_end],
                y_cls[:train_end],   y_reg[:train_end],
            )
            val_ds = TradingDataset(
                X_price[train_end:val_end], X_sent[train_end:val_end],
                y_cls[train_end:val_end],   y_reg[train_end:val_end],
            )

            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
            val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

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

            evaluator = ModelEvaluator(model)
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
        Remove tickers with too many missing bars or no indicator data.
        Logs a warning for each removed ticker.
        """
        good = []
        async with self.pool.acquire() as conn:
            for ticker in tickers:
                row = await conn.fetchrow("""
                    SELECT
                        COUNT(*)                                              AS total,
                        COUNT(*) FILTER (WHERE volume = 0 OR volume IS NULL) AS zero_vol,
                        COUNT(*) FILTER (WHERE close  IS NULL)               AS null_close
                    FROM ohlcv_data
                    WHERE ticker = $1 AND interval = $2
                """, ticker, interval)

                if not row or row["total"] < 60:
                    logger.warning(f"⚠️  Quality filter: {ticker} has <60 bars — skipped")
                    continue

                missing_pct = (row["null_close"] or 0) / row["total"]
                if missing_pct > 0.05:
                    logger.warning(
                        f"⚠️  Quality filter: {ticker} has {missing_pct:.1%} null closes — skipped"
                    )
                    continue

                good.append(ticker)

        logger.info(f"✅ Quality filter: {len(good)}/{len(tickers)} tickers passed")
        return good

    # ── Single-ticker inference ───────────────────────────────────────────────

    async def predict_ticker(self, ticker: str, interval: str = "1d") -> dict:
        from app.ml.models.registry import get_model, load_scaler_params
        from app.services.cache_service import get_cache

        # Return cached prediction if fresh (5 min TTL)
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
        X_price, X_sent, _, _, _ = await builder.build(
            [ticker], interval, sequence_len=SEQUENCE_LEN
        )

        if X_price is None or len(X_price) == 0:
            return {"error": f"Not enough data for {ticker}"}

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

        x_price = torch.tensor(X_price[-1:], dtype=torch.float32)
        x_sent  = torch.tensor(X_sent[-1:],  dtype=torch.float32)

        current_ts = datetime.now(timezone.utc)
        result     = model.predict(x_price, x_sent, current_ts=current_ts, interval=interval)

        bars_to_entry = result["bars_to_entry"][0]
        entry_time    = entry_time_from_bars(current_ts, bars_to_entry, interval)

        output = {
            "ticker":     ticker.upper(),
            "interval":   interval,
            "action":     result["action"][0],
            "confidence": round(result["confidence"][0], 4),
            "probabilities": {
                "hold": round(result["probabilities"]["hold"][0], 4),
                "buy":  round(result["probabilities"]["buy"][0],  4),
                "sell": round(result["probabilities"]["sell"][0], 4),
            },
            "entry_price":  round(result["entry_price"][0],   4),
            "stop_loss":    round(result["stop_loss"][0],      4),
            "take_profit":  round(result["take_profit"][0],    4),
            "net_profit":   round(result["net_profit"][0],     4),
            "bars_to_entry":round(result["bars_to_entry"][0],  2),
            "entry_time":   entry_time.isoformat(),
            "entry_time_label": _entry_label(entry_time, interval),
            "source":       "ml_model",
            "generated_at": current_ts.isoformat(),
        }

        # Store in cache for 5 minutes
        await cache.set(cache_key, output, ttl=300)
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

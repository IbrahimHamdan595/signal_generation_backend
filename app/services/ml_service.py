import torch
import numpy as np
import logging
import asyncpg
from torch.utils.data import DataLoader
from datetime import datetime, timezone
from typing import List

from app.ml.data.dataset import DatasetBuilder, FEATURE_COLS, SEQUENCE_LEN
from app.ml.data.entry_time import entry_time_from_bars
from app.ml.data.torch_dataset import split_dataset
from app.ml.models.fusion_model import TradingFusionModel
from app.ml.models.registry import (
    save_model_config,
    save_scaler_params,
)
from app.ml.training.trainer import Trainer
from app.ml.evaluation.evaluator import ModelEvaluator

logger = logging.getLogger(__name__)


class MLService:
    def __init__(self, pool: asyncpg.Pool):
        self.pool = pool

    async def train(
        self,
        tickers: List[str],
        interval: str = "1d",
        seq_len: int = SEQUENCE_LEN,
        epochs: int = 50,
        batch_size: int = 32,
        lr: float = 1e-3,
    ) -> dict:
        logger.info(f"📦 Building dataset for {len(tickers)} tickers...")
        builder = DatasetBuilder(self.pool)
        X_price, X_sent, y_cls, y_reg, scaler_params = await builder.build(
            tickers, interval, seq_len
        )

        train_ds, val_ds, test_ds = split_dataset(X_price, X_sent, y_cls, y_reg)
        logger.info(
            f"📊 Split: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}"
        )

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        model_config = {
            "n_features": len(FEATURE_COLS),
            "seq_len": seq_len,
            "d_model": 64,
            "n_heads": 4,
            "n_layers": 2,
            "d_ff": 256,
            "sent_input": 4,
            "sent_dim": 16,
            "mlp_hidden": 128,
            "dropout": 0.1,
        }
        model = TradingFusionModel(**model_config)
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"🧠 Model parameters: {total_params:,}")

        trainer = Trainer(model, lr=lr)
        train_results = trainer.fit(train_loader, val_loader, epochs=epochs)
        trainer.load_best()

        evaluator = ModelEvaluator(model)
        eval_results = evaluator.evaluate(test_loader)
        evaluator.save_report(eval_results)

        save_model_config(model_config)
        save_scaler_params(scaler_params)

        return {
            "status": "success",
            "dataset": {
                "total_samples": len(y_cls),
                "train": len(train_ds),
                "val": len(val_ds),
                "test": len(test_ds),
                "class_dist": {
                    "hold": int((y_cls == 0).sum()),
                    "buy": int((y_cls == 1).sum()),
                    "sell": int((y_cls == 2).sum()),
                },
            },
            "training": train_results,
            "evaluation": eval_results,
            "model_params": total_params,
            "checkpoint": train_results["checkpoint"],
        }

    async def predict_ticker(
        self,
        ticker: str,
        interval: str = "1d",
    ) -> dict:
        from app.ml.models.registry import get_model, load_scaler_params

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
            mean = np.array(scaler["mean"], dtype=np.float32)
            std = np.array(scaler["std"], dtype=np.float32)
            X_price = (X_price - mean) / (std + 1e-8)

        x_price = torch.tensor(X_price[-1:], dtype=torch.float32)
        x_sent = torch.tensor(X_sent[-1:], dtype=torch.float32)

        current_ts = datetime.now(timezone.utc)

        result = model.predict(
            x_price, x_sent, current_ts=current_ts, interval=interval
        )

        bars_to_entry = result["bars_to_entry"][0]
        entry_time = entry_time_from_bars(current_ts, bars_to_entry, interval)

        return {
            "ticker": ticker.upper(),
            "interval": interval,
            "action": result["action"][0],
            "confidence": round(result["confidence"][0], 4),
            "probabilities": {
                "hold": round(result["probabilities"]["hold"][0], 4),
                "buy": round(result["probabilities"]["buy"][0], 4),
                "sell": round(result["probabilities"]["sell"][0], 4),
            },
            "entry_price": round(result["entry_price"][0], 4),
            "stop_loss": round(result["stop_loss"][0], 4),
            "take_profit": round(result["take_profit"][0], 4),
            "net_profit": round(result["net_profit"][0], 4),
            "bars_to_entry": round(result["bars_to_entry"][0], 2),
            "entry_time": entry_time.isoformat(),
            "entry_time_label": _entry_label(entry_time, interval),
            "source": "ml_model",
            "generated_at": current_ts.isoformat(),
        }


def _entry_label(entry_time: datetime, interval: str) -> str:
    if interval == "1d":
        return f"Next trading day open — {entry_time.strftime('%A %b %d, %Y at %I:%M %p')} EST"
    elif interval == "1h":
        return f"Next candle open — {entry_time.strftime('%b %d, %Y at %H:%M')} UTC"
    return entry_time.isoformat()

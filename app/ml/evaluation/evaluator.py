import torch
import numpy as np
import json
import os
import logging
from torch.utils.data import DataLoader
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)

logger = logging.getLogger(__name__)

LABEL_NAMES = ["Hold", "Buy", "Sell"]


class ModelEvaluator:
    """
    Evaluates the TradingFusionModel on the test set.
    Computes classification + regression + trading metrics.
    """

    def __init__(self, model, device=None):
        self.model  = model
        self.device = device or torch.device("cpu")
        self.model.to(self.device)

    def evaluate(self, test_loader: DataLoader) -> dict:
        self.model.eval()

        all_preds, all_true = [], []
        all_reg_pred, all_reg_true = [], []

        with torch.no_grad():
            for x_price, x_sent, y_cls, y_reg in test_loader:
                x_price = x_price.to(self.device)
                x_sent  = x_sent.to(self.device)

                logits, regression = self.model(x_price, x_sent)

                preds = logits.argmax(dim=-1).cpu().numpy()
                all_preds.extend(preds.tolist())
                all_true.extend(y_cls.numpy().tolist())
                all_reg_pred.extend(regression.cpu().numpy().tolist())
                all_reg_true.extend(y_reg.numpy().tolist())

        all_preds    = np.array(all_preds)
        all_true     = np.array(all_true)
        all_reg_pred = np.array(all_reg_pred)
        all_reg_true = np.array(all_reg_true)

        results = {}

        # ── Classification metrics ────────────────────────────────────────────
        results["accuracy"]       = round(accuracy_score(all_true, all_preds), 4)
        results["f1_weighted"]    = round(f1_score(all_true, all_preds, average="weighted", zero_division=0), 4)
        results["f1_macro"]       = round(f1_score(all_true, all_preds, average="macro",    zero_division=0), 4)
        results["classification_report"] = classification_report(
            all_true, all_preds,
            target_names=LABEL_NAMES,
            output_dict=True,
            zero_division=0,
        )
        results["confusion_matrix"] = confusion_matrix(all_true, all_preds).tolist()

        # ── Regression metrics (per target) ───────────────────────────────────
        target_names = ["entry_price", "stop_loss", "take_profit", "net_profit"]
        reg_metrics  = {}
        for i, name in enumerate(target_names):
            rmse = float(np.sqrt(np.mean((all_reg_pred[:, i] - all_reg_true[:, i]) ** 2)))
            mae  = float(np.mean(np.abs(all_reg_pred[:, i] - all_reg_true[:, i])))
            reg_metrics[name] = {"rmse": round(rmse, 4), "mae": round(mae, 4)}
        results["regression"] = reg_metrics

        # ── Simulated trading metrics ─────────────────────────────────────────
        results["trading"] = self._trading_metrics(all_preds, all_true)

        # Log summary
        logger.info(
            f"📊 Test Accuracy: {results['accuracy']:.4f} | "
            f"F1 (weighted): {results['f1_weighted']:.4f} | "
            f"Sharpe: {results['trading']['sharpe_ratio']:.4f}"
        )
        return results

    def _trading_metrics(self, preds: np.ndarray, true: np.ndarray) -> dict:
        """
        Simulate returns based on predictions vs true labels.
        Buy(1) correct → +1 unit return, Sell(2) correct → +1 unit, wrong → -1
        """
        returns = []
        for p, t in zip(preds, true):
            if p == 0:      # Hold — no return
                continue
            if p == t:
                returns.append(1.0)   # correct direction
            else:
                returns.append(-1.0)  # wrong direction

        if not returns:
            return {"sharpe_ratio": 0.0, "win_rate": 0.0, "total_trades": 0}

        returns = np.array(returns)
        mean_r  = returns.mean()
        std_r   = returns.std() + 1e-8
        sharpe  = float(mean_r / std_r * np.sqrt(252))   # annualised
        win_rate = float((returns > 0).mean())

        return {
            "sharpe_ratio": round(sharpe,   4),
            "win_rate":     round(win_rate, 4),
            "total_trades": int(len(returns)),
            "avg_return":   round(float(mean_r), 4),
        }

    def save_report(self, results: dict, path: str = "checkpoints/eval_report.json"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"📄 Evaluation report saved → {path}")
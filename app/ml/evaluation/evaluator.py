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
    Evaluates TradingFusionModel on the test set.

    Computes:
    - Classification: accuracy, F1, confusion matrix, per-class precision/recall
    - Regression: RMSE + MAE per target (entry, SL, TP, net_profit)
    - Trading simulation: real-price % returns → Sharpe, win rate, max drawdown
    """

    def __init__(self, model, device=None):
        self.model  = model
        self.device = device or torch.device("cpu")
        self.model.to(self.device)

    def evaluate(self, test_loader: DataLoader) -> dict:
        self.model.eval()

        all_preds, all_true         = [], []
        all_reg_pred, all_reg_true  = [], []

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
        all_reg_pred = np.array(all_reg_pred)   # (N, 5)
        all_reg_true = np.array(all_reg_true)   # (N, 5)

        results = {}

        # ── Classification metrics ────────────────────────────────────────────
        results["accuracy"]    = round(accuracy_score(all_true, all_preds), 4)
        results["f1_weighted"] = round(
            f1_score(all_true, all_preds, average="weighted", zero_division=0), 4
        )
        results["f1_macro"] = round(
            f1_score(all_true, all_preds, average="macro", zero_division=0), 4
        )
        results["classification_report"] = classification_report(
            all_true, all_preds,
            target_names=LABEL_NAMES,
            output_dict=True,
            zero_division=0,
        )
        results["confusion_matrix"] = confusion_matrix(all_true, all_preds).tolist()

        # Per-class recall for quick diagnosis
        cm = np.array(results["confusion_matrix"])
        class_recall = {}
        for i, name in enumerate(LABEL_NAMES):
            row_sum = cm[i].sum()
            class_recall[name.lower()] = round(
                float(cm[i, i] / row_sum) if row_sum > 0 else 0.0, 4
            )
        results["class_recall"] = class_recall

        # ── Regression metrics (per target) ───────────────────────────────────
        target_names = ["entry_price", "stop_loss", "take_profit", "net_profit"]
        reg_metrics  = {}
        for i, name in enumerate(target_names):
            pred_i = all_reg_pred[:, i]
            true_i = all_reg_true[:, i]
            rmse = float(np.sqrt(np.mean((pred_i - true_i) ** 2)))
            mae  = float(np.mean(np.abs(pred_i - true_i)))
            reg_metrics[name] = {"rmse": round(rmse, 4), "mae": round(mae, 4)}
        results["regression"] = reg_metrics

        # ── Real-price trading simulation ─────────────────────────────────────
        results["trading"] = self._trading_metrics(
            all_preds, all_true, all_reg_true
        )

        logger.info(
            f"📊 Test Accuracy: {results['accuracy']:.4f} | "
            f"F1 (weighted): {results['f1_weighted']:.4f} | "
            f"Sharpe: {results['trading']['sharpe_ratio']:.4f} | "
            f"Win Rate: {results['trading']['win_rate']:.4f}"
        )
        logger.info(
            f"   Recall — Hold: {class_recall['hold']:.4f}  "
            f"Buy: {class_recall['buy']:.4f}  "
            f"Sell: {class_recall['sell']:.4f}"
        )
        return results

    def _trading_metrics(
        self,
        preds:    np.ndarray,   # (N,)  predicted class
        true:     np.ndarray,   # (N,)  true class
        reg_true: np.ndarray,   # (N,5) true regression — entry/SL/TP from dataset
    ) -> dict:
        """
        Simulate real-price % returns for every non-HOLD prediction.

        For each BUY/SELL signal:
          - entry_price  = reg_true[:, 0]   (actual entry)
          - take_profit  = reg_true[:, 2]   (actual TP)
          - stop_loss    = reg_true[:, 1]   (actual SL)

        Win condition: predicted direction matches true label.
          Win  → return = (TP - entry) / entry  (BUY) or (entry - TP) / entry (SELL)
          Loss → return = (SL - entry) / entry  (BUY) or (entry - SL) / entry (SELL)

        Sharpe = annualised (assumes daily bars, 252 trading days).
        Max drawdown = largest peak-to-trough cumulative-return decline.
        """
        returns      = []
        entry_prices = reg_true[:, 0]
        stop_losses  = reg_true[:, 1]
        take_profits = reg_true[:, 2]

        for i, (p, t) in enumerate(zip(preds, true)):
            if p == 0:          # HOLD — skip
                continue

            entry = entry_prices[i]
            sl    = stop_losses[i]
            tp    = take_profits[i]

            if entry <= 0:
                continue

            if p == t:          # correct direction → hit TP
                if p == 1:      # BUY
                    ret = (tp - entry) / entry
                else:           # SELL
                    ret = (entry - tp) / entry
            else:               # wrong direction → hit SL
                if p == 1:      # predicted BUY, actual was SELL/HOLD
                    ret = (sl - entry) / entry
                else:           # predicted SELL, actual was BUY/HOLD
                    ret = (entry - sl) / entry

            returns.append(float(ret))

        if not returns:
            return {
                "sharpe_ratio":  0.0,
                "win_rate":      0.0,
                "total_trades":  0,
                "avg_return":    0.0,
                "max_drawdown":  0.0,
                "total_return":  0.0,
            }

        r       = np.array(returns)
        mean_r  = float(r.mean())
        std_r   = float(r.std()) + 1e-8
        sharpe  = float(mean_r / std_r * np.sqrt(252))
        win_rate = float((r > 0).mean())

        # Cumulative return curve for max drawdown
        cum     = np.cumprod(1 + r)
        peak    = np.maximum.accumulate(cum)
        dd      = (cum - peak) / peak
        max_dd  = float(dd.min())

        total_return = float(cum[-1] - 1.0) if len(cum) > 0 else 0.0

        return {
            "sharpe_ratio":  round(sharpe,       4),
            "win_rate":      round(win_rate,      4),
            "total_trades":  int(len(returns)),
            "avg_return":    round(mean_r,        6),
            "max_drawdown":  round(max_dd,        4),
            "total_return":  round(total_return,  4),
        }

    def save_report(
        self, results: dict, path: str = "checkpoints/eval_report.json"
    ):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"📄 Evaluation report saved → {path}")

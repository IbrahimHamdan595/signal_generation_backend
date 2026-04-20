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
from app.ml.data.dataset import FEATURE_COLS

logger = logging.getLogger(__name__)

LABEL_NAMES = ["Hold", "Buy", "Sell"]


class ModelEvaluator:
    """
    Evaluates TradingFusionModel on the test set.

    Computes:
    - Classification: accuracy, F1, confusion matrix, per-class precision/recall
    - Regression: RMSE + MAE per target (% deviations from entry)
    - Trading simulation: % returns → Sharpe, win rate, max drawdown
    - Permutation feature importance: accuracy drop when each feature is shuffled
    - Drift baseline: saves test-set probability distribution for drift monitoring
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
        # Columns: [0]=entry(0.0 anchor), [1]=sl_pct, [2]=tp_pct, [3]=net_pct
        target_names = ["entry_pct", "sl_pct", "tp_pct", "net_pct"]
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

        # ── Permutation feature importance ────────────────────────────────────
        results["feature_importance"] = self._permutation_importance(
            test_loader, base_accuracy=results["accuracy"]
        )

        # ── Drift baseline — save test-set probability distribution ───────────
        self._save_drift_baseline(test_loader, results)

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
        top5 = results["feature_importance"][:5]
        logger.info(
            f"   Top-5 features: {[f['feature'] for f in top5]}"
        )
        return results

    def _permutation_importance(
        self, test_loader: DataLoader, base_accuracy: float, n_repeats: int = 3
    ) -> list:
        """
        Permutation feature importance: for each price feature, shuffle that
        column across the entire test set and measure the drop in accuracy.

        importance[i] = base_accuracy - accuracy_with_feature_i_shuffled
        Higher = more important (shuffling hurts more).

        Returns list of dicts sorted by importance descending.
        """
        n_features = len(FEATURE_COLS)

        # Collect all test batches into memory (needed for shuffling per-feature)
        all_xp, all_xs, all_yc = [], [], []
        for xp, xs, yc, _ in test_loader:
            all_xp.append(xp.numpy())
            all_xs.append(xs.numpy())
            all_yc.append(yc.numpy())

        if not all_xp:
            return []

        X_price = np.concatenate(all_xp, axis=0)   # (N, seq_len, F)
        X_sent  = np.concatenate(all_xs, axis=0)   # (N, 4)
        y_true  = np.concatenate(all_yc, axis=0)   # (N,)

        importances = []

        self.model.eval()
        with torch.no_grad():
            for feat_idx in range(n_features):
                drops = []
                for _ in range(n_repeats):
                    X_permuted = X_price.copy()
                    # Shuffle this feature across all samples (all timesteps)
                    perm_idx = np.random.permutation(len(X_permuted))
                    X_permuted[:, :, feat_idx] = X_permuted[perm_idx, :, feat_idx]

                    # Run inference in batches
                    preds = []
                    batch_size = 256
                    for start in range(0, len(X_permuted), batch_size):
                        xp_t = torch.tensor(X_permuted[start:start+batch_size], dtype=torch.float32).to(self.device)
                        xs_t = torch.tensor(X_sent[start:start+batch_size],     dtype=torch.float32).to(self.device)
                        logits, _ = self.model(xp_t, xs_t)
                        preds.extend(logits.argmax(dim=-1).cpu().numpy().tolist())

                    perm_acc = accuracy_score(y_true, preds)
                    drops.append(base_accuracy - perm_acc)

                importances.append({
                    "feature":    FEATURE_COLS[feat_idx],
                    "importance": round(float(np.mean(drops)), 5),
                    "std":        round(float(np.std(drops)),  5),
                })

        importances.sort(key=lambda x: x["importance"], reverse=True)
        return importances

    def _save_drift_baseline(self, test_loader: DataLoader, results: dict) -> None:
        """Save test-set softmax probability distributions as drift reference."""
        try:
            from app.ml.evaluation.drift_monitor import save_baseline
            prob_hold, prob_buy, prob_sell = [], [], []

            self.model.eval()
            with torch.no_grad():
                for xp, xs, _, _ in test_loader:
                    xp = xp.to(self.device)
                    xs = xs.to(self.device)
                    logits, _ = self.model(xp, xs)
                    probs = torch.softmax(logits, dim=-1).cpu().numpy()
                    prob_hold.extend(probs[:, 0].tolist())
                    prob_buy.extend( probs[:, 1].tolist())
                    prob_sell.extend(probs[:, 2].tolist())

            save_baseline(
                prob_hold=np.array(prob_hold),
                prob_buy=np.array(prob_buy),
                prob_sell=np.array(prob_sell),
                class_dist=results.get("class_recall", {}),
            )
        except Exception as e:
            logger.warning(f"⚠️  Could not save drift baseline: {e}")

    def _trading_metrics(
        self,
        preds:    np.ndarray,   # (N,)  predicted class
        true:     np.ndarray,   # (N,)  true class
        reg_true: np.ndarray,   # (N,5) true regression — % deviations from entry
    ) -> dict:
        """
        Simulate % returns for every non-HOLD prediction.

        reg_true columns (% format):
          [0] entry  = 0.0  (anchor — always zero)
          [1] sl_pct = negative % for BUY, negative % for SELL
          [2] tp_pct = positive % for BUY, positive % for SELL

        Win condition: predicted direction matches true label.
          Win  → return = tp_pct  (already a %)
          Loss → return = sl_pct  (already a %, negative)

        Sharpe = annualised (assumes daily bars, 252 trading days).
        Max drawdown = largest peak-to-trough cumulative-return decline.
        """
        returns  = []
        sl_pcts  = reg_true[:, 1]   # negative values for losses
        tp_pcts  = reg_true[:, 2]   # positive values for gains

        for i, (p, t) in enumerate(zip(preds, true)):
            if p == 0:          # HOLD — skip
                continue

            sl_pct = float(sl_pcts[i])
            tp_pct = float(tp_pcts[i])

            if p == t:          # correct direction → hit TP
                ret = abs(tp_pct)
            else:               # wrong direction → hit SL
                ret = -abs(sl_pct)

            returns.append(ret)

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

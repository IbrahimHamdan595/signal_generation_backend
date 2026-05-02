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
SENTIMENT_FEATURE_NAMES = ["avg_positive", "avg_negative", "avg_neutral", "avg_compound"]

# Approximate bars per trading year by interval — used to annualise Sharpe
# from the actual trade frequency observed in the test set.
BARS_PER_YEAR_BY_INTERVAL = {
    "1d":  252,
    "1h":  252 * 6.5,      # 6.5 trading hours/day
    "30m": 252 * 13,
    "15m": 252 * 26,
    "5m":  252 * 78,
}


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

    def __init__(self, model, device=None, interval: str = "1d"):
        self.model  = model
        self.device = device or torch.device("cpu")
        self.model.to(self.device)
        self.interval = interval

    def evaluate(self, test_loader: DataLoader) -> dict:
        self.model.eval()

        all_preds, all_true               = [], []
        all_reg_pred, all_reg_true        = [], []
        all_timing_pred, all_timing_true  = [], []

        with torch.no_grad():
            for batch in test_loader:
                # Handle both old 4-tuple (compat) and new 5-tuple datasets
                if len(batch) == 5:
                    x_price, x_sent, y_cls, y_reg, y_timing = batch
                else:
                    x_price, x_sent, y_cls, y_reg = batch
                    y_timing = None

                x_price = x_price.to(self.device)
                x_sent  = x_sent.to(self.device)

                dir_logits, regression, timing_logits = self.model(x_price, x_sent)

                preds = dir_logits.argmax(dim=-1).cpu().numpy()
                all_preds.extend(preds.tolist())
                all_true.extend(y_cls.numpy().tolist())
                all_reg_pred.extend(regression.cpu().numpy().tolist())
                all_reg_true.extend(y_reg.numpy().tolist())

                t_preds = timing_logits.argmax(dim=-1).cpu().numpy()
                all_timing_pred.extend(t_preds.tolist())
                if y_timing is not None:
                    all_timing_true.extend(y_timing.numpy().tolist())

        all_preds        = np.array(all_preds)
        all_true         = np.array(all_true)
        all_reg_pred     = np.array(all_reg_pred)     # (N, 5)
        all_reg_true     = np.array(all_reg_true)     # (N, 5)
        all_timing_pred  = np.array(all_timing_pred)  # (N,)
        all_timing_true  = np.array(all_timing_true) if all_timing_true else None

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
        target_names = ["entry_offset_pct", "sl_pct", "tp_pct", "net_pct", "bars_to_entry"]
        reg_metrics  = {}
        for i, name in enumerate(target_names):
            pred_i = all_reg_pred[:, i]
            true_i = all_reg_true[:, i]
            rmse = float(np.sqrt(np.mean((pred_i - true_i) ** 2)))
            mae  = float(np.mean(np.abs(pred_i - true_i)))
            reg_metrics[name] = {
                "rmse": round(rmse, 4),
                "mae":  round(mae,  4),
                "mae_bps": round(mae * 10000, 2) if "pct" in name else None,
            }
        results["regression"] = reg_metrics

        # ── Timing head metrics ────────────────────────────────────────────────
        if all_timing_true is not None and len(all_timing_true) == len(all_timing_pred):
            t_acc = float(accuracy_score(all_timing_true, all_timing_pred))
            results["timing"] = {
                "accuracy":          round(t_acc, 4),
                "confusion_matrix":  confusion_matrix(all_timing_true, all_timing_pred).tolist(),
            }
        else:
            results["timing"] = {"accuracy": None, "note": "no timing labels in this loader"}

        # ── Hypothetical fill rate ─────────────────────────────────────────────
        # For non-HOLD predictions: % of test samples where, in the next
        # ENTRY_WINDOW bars, price actually crossed the predicted limit entry.
        # Uses true y_reg[:, 0] as a proxy for "was there a price in that range."
        results["fill_rate"] = self._fill_rate(
            all_preds, all_reg_pred[:, 0], all_reg_true[:, 0]
        )

        # ── Real-price trading simulation ─────────────────────────────────────
        # Pass *predicted* SL/TP magnitudes so we don't leak ground-truth
        # future prices into the simulation (oracle leak fixed).
        results["trading"] = self._trading_metrics(
            all_preds, all_true, all_reg_pred, total_bars=len(all_true)
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

    def _fill_rate(
        self,
        preds:            np.ndarray,  # (N,) predicted direction
        entry_offset_pred: np.ndarray, # (N,) predicted entry % from close
        entry_offset_true: np.ndarray, # (N,) actual early-window best %
    ) -> dict:
        """
        Hypothetical fill rate: among non-HOLD predictions, how often does
        the predicted limit entry fall within the true best-price range for
        that window? A fill is counted when the predicted offset is >= the
        true best offset for BUY (model bids below or at the actual dip)
        or <= for SELL (model offers above or at the actual spike).

        This approximates "would the limit order have been filled?" without
        needing tick data.
        """
        n_non_hold = 0
        n_filled   = 0
        for i, p in enumerate(preds):
            if p == 0:  # HOLD
                continue
            n_non_hold += 1
            pred_off = float(entry_offset_pred[i])
            true_off = float(entry_offset_true[i])
            if p == 1:  # BUY — fill if predicted entry >= actual dip (limit is at or below dip)
                if pred_off >= true_off:
                    n_filled += 1
            else:       # SELL — fill if predicted entry <= actual spike
                if pred_off <= true_off:
                    n_filled += 1

        fill_rate = n_filled / n_non_hold if n_non_hold > 0 else 0.0
        return {
            "fill_rate":   round(fill_rate, 4),
            "filled":      n_filled,
            "non_hold":    n_non_hold,
        }

    def _permutation_importance(
        self, test_loader: DataLoader, base_accuracy: float, n_repeats: int = 3
    ) -> list:
        """
        Permutation feature importance: for each input feature (price *and*
        sentiment), shuffle that column across the test set and measure the
        accuracy drop.

        importance[i] = base_accuracy - accuracy_with_feature_i_shuffled
        Higher = more important (shuffling hurts more).
        """
        n_price = len(FEATURE_COLS)
        n_sent  = len(SENTIMENT_FEATURE_NAMES)

        all_xp, all_xs, all_yc = [], [], []
        for batch in test_loader:
            xp, xs, yc = batch[0], batch[1], batch[2]
            all_xp.append(xp.numpy())
            all_xs.append(xs.numpy())
            all_yc.append(yc.numpy())

        if not all_xp:
            return []

        X_price = np.concatenate(all_xp, axis=0)   # (N, seq_len, F)
        X_sent  = np.concatenate(all_xs, axis=0)   # (N, 4)
        y_true  = np.concatenate(all_yc, axis=0)   # (N,)

        def _run_inference(xp_arr: np.ndarray, xs_arr: np.ndarray) -> list:
            preds = []
            batch_size = 256
            for start in range(0, len(xp_arr), batch_size):
                xp_t = torch.tensor(xp_arr[start:start+batch_size], dtype=torch.float32).to(self.device)
                xs_t = torch.tensor(xs_arr[start:start+batch_size], dtype=torch.float32).to(self.device)
                dir_logits, _, _ = self.model(xp_t, xs_t)
                preds.extend(dir_logits.argmax(dim=-1).cpu().numpy().tolist())
            return preds

        importances = []
        self.model.eval()
        with torch.no_grad():
            # Price features
            for feat_idx in range(n_price):
                drops = []
                for _ in range(n_repeats):
                    X_perm = X_price.copy()
                    perm_idx = np.random.permutation(len(X_perm))
                    X_perm[:, :, feat_idx] = X_perm[perm_idx, :, feat_idx]
                    preds = _run_inference(X_perm, X_sent)
                    drops.append(base_accuracy - accuracy_score(y_true, preds))
                importances.append({
                    "feature":    FEATURE_COLS[feat_idx],
                    "kind":       "price",
                    "importance": round(float(np.mean(drops)), 5),
                    "std":        round(float(np.std(drops)),  5),
                })

            # Sentiment features
            for sent_idx in range(n_sent):
                drops = []
                for _ in range(n_repeats):
                    Xs_perm = X_sent.copy()
                    perm_idx = np.random.permutation(len(Xs_perm))
                    Xs_perm[:, sent_idx] = Xs_perm[perm_idx, sent_idx]
                    preds = _run_inference(X_price, Xs_perm)
                    drops.append(base_accuracy - accuracy_score(y_true, preds))
                importances.append({
                    "feature":    SENTIMENT_FEATURE_NAMES[sent_idx],
                    "kind":       "sentiment",
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
                for batch in test_loader:
                    xp = batch[0].to(self.device)
                    xs = batch[1].to(self.device)
                    dir_logits, _, _ = self.model(xp, xs)
                    probs = torch.softmax(dir_logits, dim=-1).cpu().numpy()
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
        preds:      np.ndarray,   # (N,)  predicted class
        true:       np.ndarray,   # (N,)  true class
        reg_pred:   np.ndarray,   # (N,5) predicted regression — % deviations
        total_bars: int,
    ) -> dict:
        """
        Simulate % returns for every non-HOLD prediction using the model's
        own predicted SL/TP magnitudes — never reading future ground-truth
        prices, so the result reflects what the model could have actually
        executed without an oracle.

        Win condition: predicted direction matches true label.
          Win  → return = abs(predicted_tp_pct)
          Loss → return = -abs(predicted_sl_pct)

        Sharpe annualisation factor reflects the *actual* trade frequency
        observed in the test window for the given bar interval, instead of
        always assuming one trade per daily bar.
        """
        returns  = []
        sl_pcts  = reg_pred[:, 1]   # predicted SL%
        tp_pcts  = reg_pred[:, 2]   # predicted TP%

        for i, (p, t) in enumerate(zip(preds, true)):
            if p == 0:          # HOLD — skip
                continue

            sl_pct = float(sl_pcts[i])
            tp_pct = float(tp_pcts[i])

            if p == t:
                ret = abs(tp_pct)
            else:
                ret = -abs(sl_pct)

            returns.append(ret)

        if not returns:
            return {
                "sharpe_ratio":   0.0,
                "win_rate":       0.0,
                "total_trades":   0,
                "avg_return":     0.0,
                "max_drawdown":   0.0,
                "total_return":   0.0,
                "annualisation":  {"factor": 0.0, "interval": self.interval},
            }

        r       = np.array(returns)
        mean_r  = float(r.mean())
        std_r   = float(r.std()) + 1e-8
        win_rate = float((r > 0).mean())

        # Annualisation: scale per-trade Sharpe by sqrt(trades_per_year)
        bars_per_year = BARS_PER_YEAR_BY_INTERVAL.get(self.interval, 252)
        trade_freq = (len(returns) / max(total_bars, 1)) if total_bars > 0 else 0.0
        trades_per_year = trade_freq * bars_per_year
        ann_factor = float(np.sqrt(max(trades_per_year, 1.0)))
        sharpe = float(mean_r / std_r * ann_factor)

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
            "annualisation": {
                "factor":           round(ann_factor, 4),
                "trades_per_year":  round(trades_per_year, 2),
                "bars_per_year":    bars_per_year,
                "interval":         self.interval,
            },
            "uses_predicted_levels": True,
        }

    def save_report(
        self, results: dict, path: str = "checkpoints/eval_report.json"
    ):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"📄 Evaluation report saved → {path}")

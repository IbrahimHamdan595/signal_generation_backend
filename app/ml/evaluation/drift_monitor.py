"""
Drift Monitor
=============
Detects when the model's live prediction distribution diverges from the
distribution seen during training (population stability index + KL divergence).

Saved to checkpoints/drift_baseline.json after every training run.
Checked at inference time — result exposed via GET /ml/drift.
"""

import json
import os
import numpy as np
import logging
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

BASELINE_PATH = "checkpoints/drift_baseline.json"
DRIFT_LOG_PATH = "checkpoints/drift_log.json"

# Thresholds (industry standard PSI interpretation)
PSI_STABLE    = 0.10   # < 0.10  → no significant shift
PSI_MONITOR   = 0.20   # 0.10–0.20 → monitor
PSI_RETRAIN   = 0.20   # > 0.20  → retrain recommended


def _kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-8) -> float:
    """KL(P || Q) — how much P diverges from Q (training baseline Q)."""
    p = np.clip(p, eps, 1)
    q = np.clip(q, eps, 1)
    return float(np.sum(p * np.log(p / q)))


def _psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """
    Population Stability Index for a single probability column.
    PSI = sum((actual% - expected%) * ln(actual% / expected%))
    """
    # Use fixed bins over [0, 1] for probability values
    breakpoints = np.linspace(0, 1, bins + 1)
    expected_pct = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    actual_pct   = np.histogram(actual,   bins=breakpoints)[0] / len(actual)

    eps = 1e-8
    expected_pct = np.clip(expected_pct, eps, None)
    actual_pct   = np.clip(actual_pct,   eps, None)

    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi)


def save_baseline(
    prob_hold: np.ndarray,
    prob_buy:  np.ndarray,
    prob_sell: np.ndarray,
    class_dist: dict,
) -> None:
    """
    Called after training. Saves the test-set probability distributions
    as the reference baseline for future drift checks.
    """
    baseline = {
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "n_samples": int(len(prob_hold)),
        "class_dist": class_dist,
        "mean_prob": {
            "hold": float(np.mean(prob_hold)),
            "buy":  float(np.mean(prob_buy)),
            "sell": float(np.mean(prob_sell)),
        },
        "std_prob": {
            "hold": float(np.std(prob_hold)),
            "buy":  float(np.std(prob_buy)),
            "sell": float(np.std(prob_sell)),
        },
        # Store histogram breakpoints for PSI computation
        "hist": {
            "hold": np.histogram(prob_hold, bins=10, range=(0, 1))[0].tolist(),
            "buy":  np.histogram(prob_buy,  bins=10, range=(0, 1))[0].tolist(),
            "sell": np.histogram(prob_sell, bins=10, range=(0, 1))[0].tolist(),
        },
    }
    os.makedirs(os.path.dirname(BASELINE_PATH), exist_ok=True)
    with open(BASELINE_PATH, "w") as f:
        json.dump(baseline, f, indent=2)
    logger.info(f"📊 Drift baseline saved → {BASELINE_PATH}")


def load_baseline() -> Optional[dict]:
    if not os.path.exists(BASELINE_PATH):
        return None
    with open(BASELINE_PATH) as f:
        return json.load(f)


def check_drift(
    recent_probs: list[dict],  # list of {"hold":float, "buy":float, "sell":float}
) -> dict:
    """
    Compare recent inference probabilities against the training baseline.

    recent_probs: last N predictions, each a dict with hold/buy/sell keys.
    Returns a drift report with PSI and KL divergence per class.
    """
    baseline = load_baseline()
    if baseline is None:
        return {"error": "No drift baseline found. Train the model first."}

    if len(recent_probs) < 10:
        return {"error": f"Need at least 10 recent predictions for drift check (have {len(recent_probs)})."}

    recent_hold = np.array([p["hold"] for p in recent_probs])
    recent_buy  = np.array([p["buy"]  for p in recent_probs])
    recent_sell = np.array([p["sell"] for p in recent_probs])

    # PSI per class using baseline histograms
    def _psi_from_hist(hist_expected: list, actual: np.ndarray) -> float:
        breakpoints = np.linspace(0, 1, 11)
        actual_hist = np.histogram(actual, bins=breakpoints)[0]
        exp   = np.clip(np.array(hist_expected, dtype=float), 1e-8, None)
        act   = np.clip(actual_hist.astype(float),            1e-8, None)
        exp  /= exp.sum()
        act  /= act.sum()
        return float(np.sum((act - exp) * np.log(act / exp)))

    psi_hold = _psi_from_hist(baseline["hist"]["hold"], recent_hold)
    psi_buy  = _psi_from_hist(baseline["hist"]["buy"],  recent_buy)
    psi_sell = _psi_from_hist(baseline["hist"]["sell"], recent_sell)
    psi_max  = max(psi_hold, psi_buy, psi_sell)

    # KL divergence: mean recent distribution vs mean training distribution
    mean_recent   = np.array([np.mean(recent_hold), np.mean(recent_buy), np.mean(recent_sell)])
    mean_training = np.array([
        baseline["mean_prob"]["hold"],
        baseline["mean_prob"]["buy"],
        baseline["mean_prob"]["sell"],
    ])
    mean_recent   = mean_recent   / (mean_recent.sum()   + 1e-8)
    mean_training = mean_training / (mean_training.sum() + 1e-8)
    kl = _kl_divergence(mean_recent, mean_training)

    if psi_max < PSI_STABLE:
        status = "stable"
        recommendation = "No action needed."
    elif psi_max < PSI_RETRAIN:
        status = "monitor"
        recommendation = "Mild distribution shift detected. Monitor closely."
    else:
        status = "retrain"
        recommendation = "Significant distribution shift. Retraining recommended."

    report = {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "n_recent_predictions": len(recent_probs),
        "baseline_n_samples": baseline["n_samples"],
        "baseline_saved_at": baseline["saved_at"],
        "status": status,
        "recommendation": recommendation,
        "psi": {
            "hold": round(psi_hold, 4),
            "buy":  round(psi_buy,  4),
            "sell": round(psi_sell, 4),
            "max":  round(psi_max,  4),
        },
        "kl_divergence": round(kl, 4),
        "mean_prob_recent":   {k: round(float(v), 4) for k, v in zip(["hold","buy","sell"], mean_recent)},
        "mean_prob_training": baseline["mean_prob"],
        "thresholds": {"stable": PSI_STABLE, "retrain": PSI_RETRAIN},
    }

    # Append to drift log
    _append_drift_log(report)
    return report


def _append_drift_log(report: dict) -> None:
    log = []
    if os.path.exists(DRIFT_LOG_PATH):
        try:
            with open(DRIFT_LOG_PATH) as f:
                log = json.load(f)
        except Exception:
            log = []
    log.append({
        "checked_at": report["checked_at"],
        "status": report["status"],
        "psi_max": report["psi"]["max"],
        "kl": report["kl_divergence"],
    })
    log = log[-100:]  # keep last 100 checks
    with open(DRIFT_LOG_PATH, "w") as f:
        json.dump(log, f, indent=2)

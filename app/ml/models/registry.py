import torch
import json
import os
import shutil
import logging
from datetime import datetime, timezone
from typing import Optional, List

from app.ml.models.fusion_model import TradingFusionModel

logger = logging.getLogger(__name__)

CHECKPOINT_DIR = "checkpoints"
# Canonical symlinks — always point to the current best model
MODEL_PATH     = os.path.join(CHECKPOINT_DIR, "best_model.pt")
SCALER_PATH    = os.path.join(CHECKPOINT_DIR, "scaler_params.json")
MODEL_CFG_PATH = os.path.join(CHECKPOINT_DIR, "model_config.json")
# Version history index
VERSION_INDEX  = os.path.join(CHECKPOINT_DIR, "versions.json")

# In-memory singletons
_model_instance: Optional[TradingFusionModel] = None
_scaler_params:  Optional[dict] = None


# ── Version management ────────────────────────────────────────────────────────

def _timestamp_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _load_version_index() -> List[dict]:
    if os.path.exists(VERSION_INDEX):
        with open(VERSION_INDEX) as f:
            return json.load(f)
    return []


def _save_version_index(versions: List[dict]):
    with open(VERSION_INDEX, "w") as f:
        json.dump(versions, f, indent=2)


def new_checkpoint_name() -> str:
    """Return a timestamped checkpoint filename, e.g. model_20240315_143022.pt"""
    return f"model_{_timestamp_tag()}.pt"


def register_version(
    checkpoint_name: str,
    val_loss: float,
    val_acc: float,
    eval_metrics: Optional[dict] = None,
    tickers: Optional[List[str]] = None,
) -> dict:
    """
    Record a new version in versions.json and copy it as best_model.pt
    only if it beats the current best val_loss.
    Returns the version entry.
    """
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    versions = _load_version_index()

    # Determine if this is the best model so far
    best_loss = min((v["val_loss"] for v in versions), default=float("inf"))
    is_best   = val_loss < best_loss

    entry = {
        "version":    checkpoint_name,
        "created_at": _timestamp_tag(),
        "val_loss":   round(val_loss, 6),
        "val_acc":    round(val_acc, 6),
        "is_best":    is_best,
        "tickers":    tickers or [],
        "sharpe":     round(eval_metrics.get("trading", {}).get("sharpe_ratio", 0.0), 4)
                      if eval_metrics else None,
        "accuracy":   round(eval_metrics.get("accuracy", 0.0), 4)
                      if eval_metrics else None,
    }

    # Mark previous best as no longer best
    if is_best:
        for v in versions:
            v["is_best"] = False

    versions.append(entry)
    _save_version_index(versions)

    # Promote to canonical best_model.pt
    if is_best:
        versioned_path = os.path.join(CHECKPOINT_DIR, checkpoint_name)
        if os.path.exists(versioned_path):
            shutil.copy2(versioned_path, MODEL_PATH)
            logger.info(f"🏆 New best model → {checkpoint_name} (val_loss={val_loss:.4f})")

    return entry


def list_versions() -> List[dict]:
    """Return all recorded model versions, newest first."""
    versions = _load_version_index()
    return list(reversed(versions))


def rollback_to(checkpoint_name: str) -> bool:
    """
    Promote a specific versioned checkpoint to best_model.pt.
    Returns True on success.
    """
    src = os.path.join(CHECKPOINT_DIR, checkpoint_name)
    if not os.path.exists(src):
        logger.error(f"❌ Checkpoint not found: {checkpoint_name}")
        return False

    shutil.copy2(src, MODEL_PATH)

    # Update is_best flags in index
    versions = _load_version_index()
    for v in versions:
        v["is_best"] = v["version"] == checkpoint_name
    _save_version_index(versions)

    # Reload in-memory model
    global _model_instance
    _model_instance = None
    load_model()

    logger.info(f"↩️  Rolled back to {checkpoint_name}")
    return True


# ── Config / scaler persistence ───────────────────────────────────────────────

def save_model_config(config: dict):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    with open(MODEL_CFG_PATH, "w") as f:
        json.dump(config, f, indent=2)
    logger.info(f"💾 Model config saved → {MODEL_CFG_PATH}")


def save_scaler_params(params: dict):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    with open(SCALER_PATH, "w") as f:
        json.dump(params, f, indent=2)
    logger.info(f"💾 Scaler params saved → {SCALER_PATH}")


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model() -> Optional[TradingFusionModel]:
    global _model_instance
    if not os.path.exists(MODEL_PATH) or not os.path.exists(MODEL_CFG_PATH):
        logger.warning("⚠️  No trained model found.")
        return None

    with open(MODEL_CFG_PATH) as f:
        config = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = TradingFusionModel(**config)
    ckpt   = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    _model_instance = model

    logger.info(
        f"✅ Model loaded on {device} (epoch {ckpt.get('epoch')}, "
        f"val_loss={ckpt.get('val_loss', 'N/A')})"
    )
    return model


def get_model() -> Optional[TradingFusionModel]:
    global _model_instance
    if _model_instance is None:
        _model_instance = load_model()
    return _model_instance


def load_scaler_params() -> Optional[dict]:
    global _scaler_params
    if _scaler_params is None and os.path.exists(SCALER_PATH):
        with open(SCALER_PATH) as f:
            _scaler_params = json.load(f)
    return _scaler_params


def reload_model():
    """Force reload from disk (call after rollback or new training)."""
    global _model_instance, _scaler_params
    _model_instance = None
    _scaler_params  = None
    load_model()


def is_model_trained() -> bool:
    return os.path.exists(MODEL_PATH) and os.path.exists(MODEL_CFG_PATH)

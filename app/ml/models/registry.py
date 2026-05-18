"""Per-asset-class model registry.

Each asset class (`equities`, `fx`) has its own checkpoint folder, its own
in-memory model + scaler singletons, and its own versions.json history.
All public functions default `asset_class='equities'` so existing call
sites that pre-date the FX pipeline keep working unchanged.

Layout on disk:
    checkpoints/
    ├── equities/
    │   ├── best_model.pt
    │   ├── model_config.json
    │   ├── scaler_params.json
    │   ├── model_<ts>.pt …
    │   ├── eval_report.json
    │   ├── versions.json
    │   └── …
    └── fx/
        ├── best_model.pt
        ├── …
"""

from __future__ import annotations

import torch
import json
import os
import shutil
import logging
from datetime import datetime, timezone
from typing import Optional, List

from app.ml.models.fusion_model import TradingFusionModel

logger = logging.getLogger(__name__)


# Root checkpoint directory — per-asset-class subfolders live underneath.
CHECKPOINT_ROOT = "checkpoints"

# Valid asset-class names map to subfolders of CHECKPOINT_ROOT.
_VALID_ASSET_CLASSES = ("equities", "fx")


def _normalize_asset_class(asset_class: str) -> str:
    """Accepts 'equity'/'equities'/'fx_major'/'fx_metal'/'fx' and normalises
    to one of the two checkpoint-folder names. Equity-side ingest emits
    `asset_class_for(ticker) == 'equity'` (singular); training and registry
    use the plural folder name."""
    a = (asset_class or "").lower()
    if a in ("equity", "equities"):
        return "equities"
    if a in ("fx", "fx_major", "fx_metal"):
        return "fx"
    raise ValueError(f"Unknown asset_class={asset_class!r}; expected one of {_VALID_ASSET_CLASSES} (or equity/fx_major/fx_metal)")


def _paths_for(asset_class: str) -> dict:
    ac = _normalize_asset_class(asset_class)
    base = os.path.join(CHECKPOINT_ROOT, ac)
    return {
        "asset_class":   ac,
        "dir":           base,
        "model":         os.path.join(base, "best_model.pt"),
        "scaler":        os.path.join(base, "scaler_params.json"),
        "config":        os.path.join(base, "model_config.json"),
        "version_index": os.path.join(base, "versions.json"),
        # Cloud path prefix — keeps Supabase Storage organised by asset class
        "cloud_prefix":  f"{ac}/",
    }


# In-memory singletons per asset class.
_model_instances: dict[str, Optional[TradingFusionModel]] = {"equities": None, "fx": None}
_scaler_params:   dict[str, Optional[dict]]               = {"equities": None, "fx": None}


# ── Version management ────────────────────────────────────────────────────────

def _timestamp_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _load_version_index(asset_class: str = "equities") -> List[dict]:
    p = _paths_for(asset_class)
    if os.path.exists(p["version_index"]):
        with open(p["version_index"]) as f:
            return json.load(f)
    return []


def _save_version_index(versions: List[dict], asset_class: str = "equities"):
    p = _paths_for(asset_class)
    os.makedirs(p["dir"], exist_ok=True)
    with open(p["version_index"], "w") as f:
        json.dump(versions, f, indent=2)


def new_checkpoint_name() -> str:
    """Return a timestamped checkpoint filename, e.g. model_20240315_143022.pt.
    Filename is asset-class-agnostic — the folder it ends up in distinguishes."""
    return f"model_{_timestamp_tag()}.pt"


def register_version(
    checkpoint_name: str,
    val_loss: float,
    val_acc: float,
    eval_metrics: Optional[dict] = None,
    tickers: Optional[List[str]] = None,
    asset_class: str = "equities",
) -> dict:
    """
    Record a new version in the asset-class's versions.json and promote it
    to best_model.pt unconditionally (we just finished a full retrain so the
    new checkpoint is the intended active model). Returns the version entry.
    """
    p = _paths_for(asset_class)
    os.makedirs(p["dir"], exist_ok=True)
    versions = _load_version_index(asset_class)

    # Compare against same architecture version within this asset class only —
    # cross-asset-class comparisons are meaningless.
    same_arch = [v for v in versions if v.get("model_version") == "v2"]
    best_loss = min((v["val_loss"] for v in same_arch), default=float("inf"))
    is_best   = val_loss < best_loss

    entry = {
        "version":       checkpoint_name,
        "asset_class":   p["asset_class"],
        "created_at":    _timestamp_tag(),
        "val_loss":      round(val_loss, 6),
        "val_acc":       round(val_acc, 6),
        "is_best":       is_best,
        "tickers":       tickers or [],
        "model_version": "v2",
        "sharpe":        round(eval_metrics.get("trading", {}).get("sharpe_ratio", 0.0), 4)
                         if eval_metrics else None,
        "accuracy":      round(eval_metrics.get("accuracy", 0.0), 4)
                         if eval_metrics else None,
    }

    if is_best:
        for v in versions:
            v["is_best"] = False

    versions.append(entry)
    _save_version_index(versions, asset_class)

    # Always promote — we just finished a full retrain.
    versioned_path = os.path.join(p["dir"], checkpoint_name)
    if os.path.exists(versioned_path):
        shutil.copy2(versioned_path, p["model"])
        logger.info(f"🏆 New best {p['asset_class']} model → {checkpoint_name} (val_loss={val_loss:.4f})")
        from app.services.storage_service import upload
        upload(f"{p['cloud_prefix']}best_model.pt", p["model"])
        upload(f"{p['cloud_prefix']}versions.json", p["version_index"])

    return entry


def list_versions(asset_class: str = "equities") -> List[dict]:
    """Return all recorded model versions for `asset_class`, newest first."""
    return list(reversed(_load_version_index(asset_class)))


def rollback_to(checkpoint_name: str, asset_class: str = "equities") -> bool:
    """
    Promote a specific versioned checkpoint to best_model.pt for this asset class.
    Returns True on success.
    """
    p = _paths_for(asset_class)
    src = os.path.join(p["dir"], checkpoint_name)
    if not os.path.exists(src):
        logger.error(f"❌ Checkpoint not found: {p['asset_class']}/{checkpoint_name}")
        return False

    shutil.copy2(src, p["model"])

    versions = _load_version_index(asset_class)
    for v in versions:
        v["is_best"] = v["version"] == checkpoint_name
    _save_version_index(versions, asset_class)

    _model_instances[p["asset_class"]] = None
    load_model(asset_class)

    logger.info(f"↩️  Rolled back {p['asset_class']} model to {checkpoint_name}")
    return True


# ── Config / scaler persistence ───────────────────────────────────────────────

def save_model_config(config: dict, asset_class: str = "equities"):
    p = _paths_for(asset_class)
    os.makedirs(p["dir"], exist_ok=True)
    config_out = {**config, "model_version": "v2"}
    with open(p["config"], "w") as f:
        json.dump(config_out, f, indent=2)
    logger.info(f"💾 {p['asset_class']} model config saved → {p['config']}")
    from app.services.storage_service import upload
    upload(f"{p['cloud_prefix']}model_config.json", p["config"])


def save_scaler_params(params: dict, asset_class: str = "equities"):
    p = _paths_for(asset_class)
    os.makedirs(p["dir"], exist_ok=True)
    with open(p["scaler"], "w") as f:
        json.dump(params, f, indent=2)
    logger.info(f"💾 {p['asset_class']} scaler params saved → {p['scaler']}")
    from app.services.storage_service import upload
    upload(f"{p['cloud_prefix']}scaler_params.json", p["scaler"])


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(asset_class: str = "equities") -> Optional[TradingFusionModel]:
    """Load the best model for `asset_class` into the in-memory singleton.
    Pulls missing files from Supabase Storage when configured."""
    p = _paths_for(asset_class)
    ac = p["asset_class"]

    from app.services.storage_service import download
    os.makedirs(p["dir"], exist_ok=True)
    if not os.path.exists(p["model"]):
        download(f"{p['cloud_prefix']}best_model.pt", p["model"])
    if not os.path.exists(p["config"]):
        download(f"{p['cloud_prefix']}model_config.json", p["config"])
    if not os.path.exists(p["scaler"]):
        download(f"{p['cloud_prefix']}scaler_params.json", p["scaler"])

    if not os.path.exists(p["model"]) or not os.path.exists(p["config"]):
        logger.warning(f"⚠️  No trained {ac} model found.")
        return None

    with open(p["config"]) as f:
        config = json.load(f)

    arch_version = config.get("model_version", "v1")

    # Strip inference-only keys that are stored alongside arch params but must
    # not be forwarded to the constructor (temperature, thresholds, etc.)
    _INFERENCE_KEYS = {
        "temperature", "confidence_threshold", "margin_threshold",
        "disabled_actions",
    }
    arch_config = {k: v for k, v in config.items() if k not in _INFERENCE_KEYS}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = TradingFusionModel(**arch_config)
    ckpt   = torch.load(p["model"], map_location=device, weights_only=True)
    try:
        model.load_state_dict(ckpt["model_state"])
    except RuntimeError as exc:
        logger.warning(
            f"⚠️  Saved {ac} checkpoint is incompatible with current architecture "
            f"(model_version={arch_version}, expected v2). Retrain required. "
            f"Details: {exc}"
        )
        _model_instances[ac] = None
        return None
    model.to(device)
    model.eval()
    _model_instances[ac] = model

    logger.info(
        f"✅ {ac} model loaded on {device} (epoch {ckpt.get('epoch')}, "
        f"val_loss={ckpt.get('val_loss', 'N/A')})"
    )
    return model


def get_model(asset_class: str = "equities") -> Optional[TradingFusionModel]:
    p = _paths_for(asset_class)
    ac = p["asset_class"]
    if _model_instances.get(ac) is None:
        _model_instances[ac] = load_model(ac)
    return _model_instances[ac]


def load_scaler_params(asset_class: str = "equities") -> Optional[dict]:
    p = _paths_for(asset_class)
    ac = p["asset_class"]
    if _scaler_params.get(ac) is None and os.path.exists(p["scaler"]):
        with open(p["scaler"]) as f:
            _scaler_params[ac] = json.load(f)
    return _scaler_params.get(ac)


def reload_model(asset_class: str = "equities"):
    """Force reload from disk (call after rollback or new training)."""
    p = _paths_for(asset_class)
    ac = p["asset_class"]
    _model_instances[ac] = None
    _scaler_params[ac]   = None
    load_model(ac)


def is_model_trained(asset_class: str = "equities") -> bool:
    p = _paths_for(asset_class)
    return os.path.exists(p["model"]) and os.path.exists(p["config"])


# ── Backwards-compatible aliases used by older code paths ─────────────────────
# Some legacy modules import CHECKPOINT_DIR / MODEL_PATH directly. Keep them
# resolving to the equities namespace so nothing breaks during the rollout.

CHECKPOINT_DIR = os.path.join(CHECKPOINT_ROOT, "equities")
MODEL_PATH     = os.path.join(CHECKPOINT_DIR, "best_model.pt")
SCALER_PATH    = os.path.join(CHECKPOINT_DIR, "scaler_params.json")
MODEL_CFG_PATH = os.path.join(CHECKPOINT_DIR, "model_config.json")
VERSION_INDEX  = os.path.join(CHECKPOINT_DIR, "versions.json")

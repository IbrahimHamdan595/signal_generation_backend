import torch
import json
import os
import logging
from typing import Optional

from app.ml.models.fusion_model import TradingFusionModel

logger = logging.getLogger(__name__)

CHECKPOINT_DIR  = "checkpoints"
MODEL_PATH      = os.path.join(CHECKPOINT_DIR, "best_model.pt")
SCALER_PATH     = os.path.join(CHECKPOINT_DIR, "scaler_params.json")
MODEL_CFG_PATH  = os.path.join(CHECKPOINT_DIR, "model_config.json")

# Singleton — loaded once when the FastAPI server starts
_model_instance: Optional[TradingFusionModel] = None
_scaler_params:  Optional[dict] = None


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


def load_model() -> Optional[TradingFusionModel]:
    """Load the best checkpoint into memory. Returns None if not trained yet."""
    global _model_instance

    if not os.path.exists(MODEL_PATH):
        logger.warning("⚠️  No trained model found. Train the model first.")
        return None

    if not os.path.exists(MODEL_CFG_PATH):
        logger.warning("⚠️  No model config found.")
        return None

    with open(MODEL_CFG_PATH) as f:
        config = json.load(f)

    model = TradingFusionModel(**config)
    ckpt  = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    _model_instance = model
    logger.info(
        f"✅ Model loaded from checkpoint "
        f"(epoch {ckpt.get('epoch')}, val_loss={ckpt.get('val_loss', 'N/A')})"
    )
    return model


def get_model() -> Optional[TradingFusionModel]:
    """Return cached model instance, loading from disk if necessary."""
    global _model_instance
    if _model_instance is None:
        _model_instance = load_model()
    return _model_instance


def load_scaler_params() -> Optional[dict]:
    """Return cached scaler params, loading from disk if necessary."""
    global _scaler_params
    if _scaler_params is None and os.path.exists(SCALER_PATH):
        with open(SCALER_PATH) as f:
            _scaler_params = json.load(f)
    return _scaler_params


def is_model_trained() -> bool:
    return os.path.exists(MODEL_PATH) and os.path.exists(MODEL_CFG_PATH)
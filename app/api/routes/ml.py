import math
import os
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Request
from slowapi import Limiter
from slowapi.util import get_remote_address
from pydantic import BaseModel, Field
from typing import Any, List, Optional

from app.core.security import get_current_active_user

limiter = Limiter(key_func=get_remote_address)
from app.db.database import get_db
from app.services.ml_service import MLService
from app.services.job_service import JobService
from app.ml.models.registry import is_model_trained
from app.ml.data.dataset import SEQUENCE_LEN

router = APIRouter(prefix="/ml", tags=["ML Model"])


def _sanitize_json(obj: Any) -> Any:
    """Recursively replace NaN/Inf floats with None so FastAPI can serialize.

    Python's `json.load` happily parses `NaN`/`Infinity` literals into floats,
    but the default `json.dumps` rejects them with
    `ValueError: Out of range float values are not JSON compliant`. Training
    code can emit NaN for undefined metrics (zero-variance Sharpe, regression
    on a single-class fold, etc.), so we coerce them to None before returning.
    """
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, dict):
        return {k: _sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_json(v) for v in obj]
    return obj


class TrainRequest(BaseModel):
    tickers: List[str] = Field(..., description="Tickers to train on")
    interval: str = Field(default="1d")
    epochs: int = Field(default=100, ge=1, le=500)
    batch_size: int = Field(default=64, ge=8, le=256)
    lr: float = Field(default=2e-4, gt=0)
    seq_len: int = Field(default=SEQUENCE_LEN, ge=10, le=200)
    use_class_weights: bool = Field(
        default=True,
        description=(
            "Use inverse-frequency class weighting to fight HOLD dominance. "
            "Disable (false) if BUY recall is collapsing — sometimes letting "
            "the data's natural imbalance through gives a more confident model."
        ),
    )
    # ── Asset class + label overrides for the multi-pipeline setup ────────────
    # asset_class selects which per-class checkpoint folder the model + its
    # config/scaler land in (`checkpoints/equities/` vs `checkpoints/fx/`).
    # The three threshold overrides let FX runs use tighter barriers than
    # equities — daily FX volatility is much smaller, so the equity 2% / 15-bar
    # labels would produce HOLD-only datasets.
    asset_class: str = Field(
        default="equities",
        description="Checkpoint namespace: 'equities' or 'fx'. Controls where the model + config get saved.",
    )
    buy_threshold: Optional[float] = Field(
        default=None, gt=0, lt=1,
        description="Override config.BUY_THRESHOLD for this training run (e.g. 0.012 for FX).",
    )
    sell_threshold: Optional[float] = Field(
        default=None, gt=0, lt=1,
        description="Override config.SELL_THRESHOLD for this training run.",
    )
    lookahead_window: Optional[int] = Field(
        default=None, ge=1, le=60,
        description="Override config.LOOKAHEAD_WINDOW (bars) for this training run.",
    )
    barrier_atr_mult: Optional[float] = Field(
        default=None, gt=0, le=10,
        description=(
            "When set, switches to volatility-normalised labelling: each "
            "bar's barrier becomes `mult × ATR(t) / close(t)`. Recommended "
            "for FX/metals where per-pair daily volatility ranges from "
            "0.3% (USDHKD) to 1.5%+ (XAUUSD). Typical value: 1.5."
        ),
    )


class PredictRequest(BaseModel):
    ticker: str
    interval: str = "1d"


@router.get("/status")
async def model_status(
    asset_class: str = "equities",
    current_user=Depends(get_current_active_user),
):
    import os
    import json
    from app.ml.models.registry import _paths_for, _normalize_asset_class

    ac = _normalize_asset_class(asset_class)
    paths = _paths_for(ac)

    trained = is_model_trained(ac)
    info = {"asset_class": ac}

    if trained:
        import torch
        ckpt = torch.load(paths["model"], map_location="cpu", weights_only=True)
        info["epoch"] = ckpt.get("epoch")
        info["val_loss"] = ckpt.get("val_loss")
        info["val_acc"] = ckpt.get("val_acc")

        eval_path = os.path.join(paths["dir"], "eval_report.json")
        if os.path.exists(eval_path):
            with open(eval_path) as f:
                report = json.load(f)
            info["test_accuracy"] = report.get("accuracy")
            info["test_f1_weighted"] = report.get("f1_weighted")
            info["sharpe_ratio"] = report.get("trading", {}).get("sharpe_ratio")

    return {"trained": trained, "info": info}


@router.post("/train")
@limiter.limit("5/hour")
async def train_model(
    request: Request,
    body: TrainRequest,
    background_tasks: BackgroundTasks,
    pool=Depends(get_db),
    current_user=Depends(get_current_active_user),
    diagnostics: bool = False,
):
    """
    Train a fresh model. When `diagnostics=true` (query param), also runs the
    embedded 3-fold walk-forward and permutation feature-importance. These
    add ~20 minutes; default fast mode skips them.
    """
    if len(body.tickers) < 1:
        raise HTTPException(400, "Provide at least 1 ticker for training")
    if len(body.tickers) > 100:
        raise HTTPException(400, "Max 100 tickers per training run")

    job_svc = JobService(pool)
    job_id = await job_svc.create("training")

    async def _run():
        svc = MLService(pool)
        _job_svc = JobService(pool)
        try:
            result = await svc.train(
                tickers=body.tickers,
                interval=body.interval,
                seq_len=body.seq_len,
                epochs=body.epochs,
                batch_size=body.batch_size,
                lr=body.lr,
                use_class_weights=body.use_class_weights,
                diagnostics=diagnostics,
                asset_class=body.asset_class,
                buy_threshold=body.buy_threshold,
                sell_threshold=body.sell_threshold,
                lookahead_window=body.lookahead_window,
                barrier_atr_mult=body.barrier_atr_mult,
            )
            import json
            import os as _os

            # Persist last_train_result.json into the asset-class subfolder
            # so equities and fx runs don't trample each other.
            ac = (body.asset_class or "equities").lower()
            ac = "equities" if ac in ("equity", "equities") else "fx"
            out_dir = _os.path.join("checkpoints", ac)
            _os.makedirs(out_dir, exist_ok=True)
            with open(_os.path.join(out_dir, "last_train_result.json"), "w") as f:
                json.dump(result, f, indent=2, default=str)
            await _job_svc.complete(job_id, {
                "tickers": len(body.tickers),
                "epochs": body.epochs,
                "status": "completed",
            })
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(f"Training failed: {e}")
            await _job_svc.fail(job_id, str(e))

    background_tasks.add_task(_run)
    return {
        "message": f"Training started for {len(body.tickers)} tickers",
        "tickers": body.tickers,
        "epochs": body.epochs,
        "job_id": job_id,
        "note": f"Poll GET /api/v1/jobs/{job_id} to monitor progress",
    }


@router.post("/train/sync")
async def train_model_sync(
    body: TrainRequest,
    pool=Depends(get_db),
    current_user=Depends(get_current_active_user),
    diagnostics: bool = False,
):
    if len(body.tickers) > 20:
        raise HTTPException(400, "Use /ml/train (background) for > 20 tickers")

    svc = MLService(pool)
    result = await svc.train(
        tickers=body.tickers,
        interval=body.interval,
        seq_len=body.seq_len,
        epochs=body.epochs,
        batch_size=body.batch_size,
        lr=body.lr,
        use_class_weights=body.use_class_weights,
        diagnostics=diagnostics,
        asset_class=body.asset_class,
        buy_threshold=body.buy_threshold,
        sell_threshold=body.sell_threshold,
        lookahead_window=body.lookahead_window,
        barrier_atr_mult=body.barrier_atr_mult,
    )
    return result


@router.post("/predict")
async def predict(
    body: PredictRequest,
    pool=Depends(get_db),
    current_user=Depends(get_current_active_user),
):
    if not is_model_trained():
        raise HTTPException(
            400, "No trained model found. Run POST /api/v1/ml/train first."
        )

    svc = MLService(pool)
    result = await svc.predict_ticker(body.ticker, body.interval)

    if "error" in result:
        raise HTTPException(404, result["error"])

    return result


@router.get("/report")
async def get_eval_report(
    asset_class: str = "equities",
    current_user=Depends(get_current_active_user),
):
    import os
    import json
    from app.ml.models.registry import _paths_for, _normalize_asset_class

    ac = _normalize_asset_class(asset_class)
    path = os.path.join(_paths_for(ac)["dir"], "eval_report.json")
    if not os.path.exists(path):
        raise HTTPException(404, f"No evaluation report for {ac}. Train the model first.")
    with open(path) as f:
        return _sanitize_json(json.load(f))


@router.get("/train/result")
async def get_last_train_result(
    asset_class: str = "equities",
    current_user=Depends(get_current_active_user),
):
    import os
    import json
    from app.ml.models.registry import _paths_for, _normalize_asset_class

    ac = _normalize_asset_class(asset_class)
    path = os.path.join(_paths_for(ac)["dir"], "last_train_result.json")
    if not os.path.exists(path):
        raise HTTPException(404, f"No training result for {ac} yet.")
    with open(path) as f:
        return _sanitize_json(json.load(f))


# ── Walk-forward validation ───────────────────────────────────────────────────

class WalkForwardRequest(BaseModel):
    tickers:         List[str] = Field(..., description="Tickers to validate on")
    interval:        str       = Field(default="1d")
    n_splits:        int       = Field(default=5, ge=2, le=10)
    epochs:          int       = Field(default=30, ge=1, le=200)
    batch_size:      int       = Field(default=32, ge=8, le=256)
    lr:              float     = Field(default=1e-4, gt=0)


@router.post("/walkforward")
async def walk_forward(
    body: WalkForwardRequest,
    background_tasks: BackgroundTasks,
    pool=Depends(get_db),
    current_user=Depends(get_current_active_user),
):
    """
    Run expanding-window walk-forward validation across n_splits folds.
    Trains a fresh model per fold and reports per-fold + aggregate metrics.
    Runs in background — poll GET /ml/walkforward/result for completion.
    """
    if len(body.tickers) < 2:
        raise HTTPException(400, "Provide at least 2 tickers")

    async def _run():
        import json
        svc = MLService(pool)
        try:
            result = await svc.walk_forward_validate(
                tickers=body.tickers,
                interval=body.interval,
                n_splits=body.n_splits,
                epochs=body.epochs,
                batch_size=body.batch_size,
                lr=body.lr,
            )
            os.makedirs("checkpoints", exist_ok=True)
            with open("checkpoints/walkforward_result.json", "w") as f:
                json.dump(result, f, indent=2, default=str)
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(f"Walk-forward failed: {e}")

    background_tasks.add_task(_run)
    return {
        "message": f"Walk-forward started ({body.n_splits} folds, {len(body.tickers)} tickers)",
        "note": "Poll GET /api/v1/ml/walkforward/result when complete",
    }


@router.get("/walkforward/result")
async def get_walkforward_result(current_user=Depends(get_current_active_user)):
    import json
    path = "checkpoints/walkforward_result.json"
    if not os.path.exists(path):
        raise HTTPException(404, "No walk-forward result yet. Run POST /ml/walkforward first.")
    with open(path) as f:
        return _sanitize_json(json.load(f))


# ── Drift monitoring ──────────────────────────────────────────────────────────

@router.get("/drift")
async def check_drift(current_user=Depends(get_current_active_user)):
    """
    Compare the recent signal prediction distribution against the training baseline.
    Uses the last 50 signals stored in the database.
    Returns PSI and KL divergence — status: stable / monitor / retrain.
    """
    from app.ml.evaluation.drift_monitor import check_drift as _check, load_baseline
    import json

    baseline = load_baseline()
    if baseline is None:
        raise HTTPException(404, "No drift baseline found. Train the model first.")

    # Load recent signal probabilities from the walkforward result or eval report
    # as a proxy — real deployment would pull from the signals table
    signal_probs_path = "checkpoints/recent_signal_probs.json"
    if not os.path.exists(signal_probs_path):
        raise HTTPException(
            404,
            "No recent signal probabilities recorded yet. "
            "Generate signals first (POST /signals/generate/batch), "
            "then re-check drift."
        )
    with open(signal_probs_path) as f:
        recent_probs = json.load(f)

    return _check(recent_probs)


@router.get("/drift/baseline")
async def get_drift_baseline(current_user=Depends(get_current_active_user)):
    """Return the saved training distribution baseline used for drift detection."""
    from app.ml.evaluation.drift_monitor import load_baseline
    baseline = load_baseline()
    if baseline is None:
        raise HTTPException(404, "No drift baseline found. Train the model first.")
    return baseline


# ── Model versioning ──────────────────────────────────────────────────────────

@router.get("/versions")
async def list_versions(
    asset_class: str = "equities",
    current_user=Depends(get_current_active_user),
):
    """List all trained model versions for one asset class, newest first."""
    from app.ml.models.registry import list_versions as _list
    return {"versions": _list(asset_class)}


@router.post("/versions/rollback")
async def rollback_version(
    version: str,
    asset_class: str = "equities",
    current_user=Depends(get_current_active_user),
):
    """
    Promote a previous checkpoint to best_model.pt for the given asset class.
    Pass the checkpoint filename, e.g. model_20240315_143022.pt
    """
    from app.ml.models.registry import rollback_to
    ok = rollback_to(version, asset_class)
    if not ok:
        raise HTTPException(404, f"Checkpoint '{version}' not found for {asset_class}")
    return {"message": f"Rolled back {asset_class} to {version}", "active": version}

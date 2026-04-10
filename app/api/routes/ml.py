import os
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Request
from slowapi import Limiter
from slowapi.util import get_remote_address
from pydantic import BaseModel, Field
from typing import List

from app.core.security import get_current_active_user

limiter = Limiter(key_func=get_remote_address)
from app.db.database import get_db
from app.services.ml_service import MLService
from app.services.job_service import JobService
from app.ml.models.registry import is_model_trained
from app.ml.data.dataset import SEQUENCE_LEN

router = APIRouter(prefix="/ml", tags=["ML Model"])


class TrainRequest(BaseModel):
    tickers: List[str] = Field(..., description="Tickers to train on")
    interval: str = Field(default="1d")
    epochs: int = Field(default=50, ge=1, le=300)
    batch_size: int = Field(default=32, ge=8, le=256)
    lr: float = Field(default=1e-3, gt=0)
    seq_len: int = Field(default=SEQUENCE_LEN, ge=10, le=200)


class PredictRequest(BaseModel):
    ticker: str
    interval: str = "1d"


@router.get("/status")
async def model_status(current_user=Depends(get_current_active_user)):
    import os
    import json

    trained = is_model_trained()
    info = {}

    if trained:
        ckpt_path = "checkpoints/best_model.pt"
        eval_path = "checkpoints/eval_report.json"

        import torch

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        info["epoch"] = ckpt.get("epoch")
        info["val_loss"] = ckpt.get("val_loss")
        info["val_acc"] = ckpt.get("val_acc")

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
):
    if len(body.tickers) < 2:
        raise HTTPException(400, "Provide at least 2 tickers for training")
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
            )
            import json

            with open("checkpoints/last_train_result.json", "w") as f:
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
async def get_eval_report(current_user=Depends(get_current_active_user)):
    import os
    import json

    path = "checkpoints/eval_report.json"
    if not os.path.exists(path):
        raise HTTPException(404, "No evaluation report found. Train the model first.")
    with open(path) as f:
        return json.load(f)


@router.get("/train/result")
async def get_last_train_result(current_user=Depends(get_current_active_user)):
    import os
    import json

    path = "checkpoints/last_train_result.json"
    if not os.path.exists(path):
        raise HTTPException(404, "No training result found yet.")
    with open(path) as f:
        return json.load(f)


# ── Walk-forward validation ───────────────────────────────────────────────────

class WalkForwardRequest(BaseModel):
    tickers:         List[str] = Field(..., description="Tickers to validate on")
    interval:        str       = Field(default="1d")
    n_splits:        int       = Field(default=5, ge=2, le=10)
    epochs:          int       = Field(default=30, ge=1, le=200)
    batch_size:      int       = Field(default=32, ge=8, le=256)
    lr:              float     = Field(default=1e-3, gt=0)


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
        return json.load(f)


# ── Model versioning ──────────────────────────────────────────────────────────

@router.get("/versions")
async def list_versions(current_user=Depends(get_current_active_user)):
    """List all trained model versions, newest first."""
    from app.ml.models.registry import list_versions as _list
    return {"versions": _list()}


@router.post("/versions/rollback")
async def rollback_version(
    version: str,
    current_user=Depends(get_current_active_user),
):
    """
    Promote a previous checkpoint to best_model.pt.
    Pass the checkpoint filename, e.g. model_20240315_143022.pt
    """
    from app.ml.models.registry import rollback_to
    ok = rollback_to(version)
    if not ok:
        raise HTTPException(404, f"Checkpoint '{version}' not found in checkpoints/")
    return {"message": f"Rolled back to {version}", "active": version}

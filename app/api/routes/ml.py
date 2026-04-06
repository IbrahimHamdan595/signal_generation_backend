from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List

from app.core.security import get_current_active_user
from app.db.database import get_db
from app.services.ml_service import MLService
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
async def train_model(
    body: TrainRequest,
    background_tasks: BackgroundTasks,
    pool=Depends(get_db),
    current_user=Depends(get_current_active_user),
):
    if len(body.tickers) < 2:
        raise HTTPException(400, "Provide at least 2 tickers for training")
    if len(body.tickers) > 100:
        raise HTTPException(400, "Max 100 tickers per training run")

    async def _run():
        svc = MLService(pool)
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
        except Exception as e:
            import logging

            logging.getLogger(__name__).error(f"Training failed: {e}")

    background_tasks.add_task(_run)
    return {
        "message": f"Training started for {len(body.tickers)} tickers",
        "tickers": body.tickers,
        "epochs": body.epochs,
        "note": "Poll GET /api/v1/ml/status to monitor progress",
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

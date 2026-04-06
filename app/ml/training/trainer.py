import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Tuple
import os
import logging

from app.ml.models.fusion_model import TradingFusionModel

logger = logging.getLogger(__name__)

CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


class Trainer:
    """
    Trains the TradingFusionModel with:
    - Joint loss: CrossEntropy (classification) + MSE (regression)
    - Early stopping on validation loss
    - Model checkpoint saving
    - CPU-optimised settings
    """

    def __init__(
        self,
        model: TradingFusionModel,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        cls_weight: float = 1.0,  # weight for classification loss
        reg_weight: float = 0.1,  # weight for regression loss
        patience: int = 10,  # early stopping patience
        checkpoint_name: str = "best_model.pt",
    ):
        self.model = model
        self.device = torch.device("cpu")
        self.model.to(self.device)

        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=5, factor=0.5, verbose=True
        )
        self.cls_loss_fn = nn.CrossEntropyLoss()
        self.reg_loss_fn = nn.MSELoss()

        self.cls_weight = cls_weight
        self.reg_weight = reg_weight
        self.patience = patience
        self.checkpoint_path = os.path.join(CHECKPOINT_DIR, checkpoint_name)

        self.train_history = []
        self.val_history = []

    # ── Main train loop ───────────────────────────────────────────────────────

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 50,
    ) -> dict:
        best_val_loss = float("inf")
        epochs_no_improve = 0

        logger.info(f"🚀 Training on CPU for {epochs} epochs...")

        for epoch in range(1, epochs + 1):
            train_loss, train_acc = self._run_epoch(train_loader, training=True)
            val_loss, val_acc = self._run_epoch(val_loader, training=False)

            self.scheduler.step(val_loss)
            self.train_history.append({"loss": train_loss, "acc": train_acc})
            self.val_history.append({"loss": val_loss, "acc": val_acc})

            logger.info(
                f"Epoch {epoch:03d}/{epochs} | "
                f"Train loss: {train_loss:.4f} acc: {train_acc:.3f} | "
                f"Val loss: {val_loss:.4f} acc: {val_acc:.3f}"
            )

            # Checkpoint best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                self._save_checkpoint(epoch, val_loss, val_acc)
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.patience:
                    logger.info(f"⏹ Early stopping at epoch {epoch}")
                    break

        logger.info(f"✅ Training done. Best val loss: {best_val_loss:.4f}")
        return {
            "best_val_loss": best_val_loss,
            "train_history": self.train_history,
            "val_history": self.val_history,
            "checkpoint": self.checkpoint_path,
        }

    # ── Epoch runner ──────────────────────────────────────────────────────────

    def _run_epoch(self, loader: DataLoader, training: bool) -> Tuple[float, float]:
        self.model.train() if training else self.model.eval()

        total_loss, correct, total = 0.0, 0, 0

        ctx = torch.enable_grad() if training else torch.no_grad()
        with ctx:
            for x_price, x_sent, y_cls, y_reg in loader:
                x_price = x_price.to(self.device)
                x_sent = x_sent.to(self.device)
                y_cls = y_cls.to(self.device)
                y_reg = y_reg.to(self.device)

                logits, regression = self.model(x_price, x_sent)

                # Joint loss
                loss = self.cls_weight * self.cls_loss_fn(
                    logits, y_cls
                ) + self.reg_weight * self.reg_loss_fn(regression, y_reg)

                if training:
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()

                total_loss += loss.item() * len(y_cls)
                preds = logits.argmax(dim=-1)
                correct += (preds == y_cls).sum().item()
                total += len(y_cls)

        return total_loss / max(total, 1), correct / max(total, 1)

    # ── Checkpoint ────────────────────────────────────────────────────────────

    def _save_checkpoint(self, epoch: int, val_loss: float, val_acc: float):
        torch.save(
            {
                "epoch": epoch,
                "model_state": self.model.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "optimizer": self.optimizer.state_dict(),
            },
            self.checkpoint_path,
        )
        logger.info(f"💾 Checkpoint saved → {self.checkpoint_path}")

    def load_best(self):
        ckpt = torch.load(self.checkpoint_path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(ckpt["model_state"])
        logger.info(
            f"✅ Loaded best checkpoint (epoch {ckpt['epoch']}, "
            f"val_loss={ckpt['val_loss']:.4f})"
        )

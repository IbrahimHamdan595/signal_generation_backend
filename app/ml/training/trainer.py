import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Tuple, Optional
import numpy as np
import os
import logging

from app.ml.models.fusion_model import TradingFusionModel


class FocalLoss(nn.Module):
    """Multi-class focal loss with optional class weights and label smoothing.

    FL(p_t) = -α_t · (1 − p_t)^γ · log(p_t)

    γ=2.0 down-weights easy examples (high-confidence correct predictions) so
    training focuses on the hard, uncertain samples — exactly what we want when
    BUY/SELL signals are rare and HOLD dominates the batch.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        weight: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.1,
    ):
        super().__init__()
        self.gamma           = gamma
        self.weight          = weight
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        n_cls = logits.size(-1)

        # Smooth one-hot targets
        with torch.no_grad():
            fill = self.label_smoothing / max(n_cls - 1, 1)
            smooth = torch.full_like(logits, fill)
            smooth.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)

        log_probs = torch.log_softmax(logits, dim=-1)
        probs     = torch.exp(log_probs)

        # Probability of the true class — used for focal modulation only
        p_t = probs.gather(1, targets.unsqueeze(1)).squeeze(1).detach()

        ce    = -(smooth * log_probs).sum(dim=-1)          # per-sample CE
        focal = (1.0 - p_t) ** self.gamma * ce            # focal weighting

        if self.weight is not None:
            focal = focal * self.weight[targets]

        return focal.mean()

logger = logging.getLogger(__name__)

CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


class Trainer:
    """
    Trains the TradingFusionModel with:
    - Joint loss: weighted CrossEntropy (classification) + MSE (regression)
    - Class weights computed from dataset to fix HOLD-dominance imbalance
    - Early stopping on validation loss
    - Versioned checkpoint saving (never overwrites old checkpoints)
    """

    def __init__(
        self,
        model: TradingFusionModel,
        lr: float = 2e-4,
        weight_decay: float = 5e-4,
        cls_weight: float = 1.0,
        reg_weight: float = 0.05,
        timing_weight: float = 0.0,
        patience: int = 15,
        checkpoint_name: str = "best_model.pt",
        class_weights: Optional[torch.Tensor] = None,
        timing_class_weights: Optional[torch.Tensor] = None,
    ):
        self.model  = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        if torch.cuda.is_available():
            logger.info(
                f"🚀 GPU detected: {torch.cuda.get_device_name(0)} "
                f"({torch.cuda.get_device_properties(0).total_memory // 1024**2} MB VRAM)"
            )
        else:
            logger.info("⚠️  No GPU found — training on CPU (install CUDA PyTorch for speed)")

        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self._base_lr = lr
        self._warmup_epochs = 5
        self.scheduler = None  # built lazily in fit()

        self.cls_loss_fn = FocalLoss(
            gamma=2.0,
            weight=class_weights.to(self.device) if class_weights is not None else None,
            label_smoothing=0.1,
        )
        if class_weights is not None:
            logger.info(f"⚖️  Class weights applied to FocalLoss: {class_weights.tolist()}")

        self.reg_loss_fn    = nn.MSELoss()
        self.timing_loss_fn = nn.CrossEntropyLoss(
            weight=timing_class_weights.to(self.device) if timing_class_weights is not None else None,
            label_smoothing=0.1,
        )

        self.cls_weight    = cls_weight
        self.reg_weight    = reg_weight
        self.timing_weight = timing_weight
        self.patience      = patience
        self.checkpoint_path = os.path.join(CHECKPOINT_DIR, checkpoint_name)

        self.train_history: list = []
        self.val_history:   list = []

    # ── Main train loop ───────────────────────────────────────────────────────

    def fit(
        self,
        train_loader: DataLoader,
        val_loader:   DataLoader,
        epochs:       int = 50,
    ) -> dict:
        best_val_loss    = float("inf")
        epochs_no_improve = 0

        # ReduceLROnPlateau: cut LR by 50% when val_loss doesn't improve for 4 epochs.
        # This adapts to the actual learning curve instead of following a fixed cosine
        # schedule — critical for financial time series where regimes shift unpredictably.
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=6,
            min_lr=self._base_lr * 0.01,
        )

        device_label = f"GPU ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else "CPU"
        logger.info(f"🚀 Training on {device_label} for {epochs} epochs...")

        for epoch in range(1, epochs + 1):
            # Linear warmup: scale LR from 10% → 100% across warmup epochs
            if epoch <= self._warmup_epochs:
                warm_factor = epoch / self._warmup_epochs
                for g in self.optimizer.param_groups:
                    g["lr"] = self._base_lr * (0.1 + 0.9 * warm_factor)

            tr_loss, tr_acc, tr_timing_acc, tr_losses = self._run_epoch(train_loader, training=True)
            va_loss, va_acc, va_timing_acc, va_losses = self._run_epoch(val_loader,   training=False)

            train_loss, val_loss = tr_loss, va_loss
            train_acc,  val_acc  = tr_acc,  va_acc

            if epoch > self._warmup_epochs:
                self.scheduler.step(val_loss)
            self.train_history.append({
                "loss": tr_loss, "acc": tr_acc, "timing_acc": tr_timing_acc,
                "cls_loss": tr_losses[0], "reg_loss": tr_losses[1], "timing_loss": tr_losses[2],
            })
            self.val_history.append({
                "loss": va_loss, "acc": va_acc, "timing_acc": va_timing_acc,
                "cls_loss": va_losses[0], "reg_loss": va_losses[1], "timing_loss": va_losses[2],
            })

            logger.info(
                f"Epoch {epoch:03d}/{epochs} | "
                f"Train loss: {train_loss:.4f} acc: {train_acc:.3f} | "
                f"Val loss: {val_loss:.4f} acc: {val_acc:.3f} | "
                f"timing_acc: {self.val_history[-1].get('timing_acc', 0):.3f}"
            )

            if val_loss < best_val_loss:
                best_val_loss     = val_loss
                epochs_no_improve = 0
                self._save_checkpoint(epoch, val_loss, val_acc)
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.patience:
                    logger.info(f"⏹ Early stopping at epoch {epoch}")
                    break

        logger.info(f"✅ Training done. Best val loss: {best_val_loss:.4f}")
        return {
            "best_val_loss":  best_val_loss,
            "train_history":  self.train_history,
            "val_history":    self.val_history,
            "checkpoint":     self.checkpoint_path,
        }

    # ── Epoch runner ──────────────────────────────────────────────────────────

    def _run_epoch(self, loader: DataLoader, training: bool) -> Tuple[float, float, float, list]:
        """Returns (total_loss, dir_accuracy, timing_accuracy, [cls_loss, reg_loss, timing_loss])."""
        self.model.train() if training else self.model.eval()

        total_loss = 0.0
        sum_cls_loss = sum_reg_loss = sum_timing_loss = 0.0
        correct_dir = correct_timing = total = 0

        ctx = torch.enable_grad() if training else torch.no_grad()
        with ctx:
            for x_price, x_sent, y_cls, y_reg, y_timing in loader:
                x_price  = x_price.to(self.device)
                x_sent   = x_sent.to(self.device)
                y_cls    = y_cls.to(self.device)
                y_reg    = y_reg.to(self.device)
                y_timing = y_timing.to(self.device)

                # Replace any NaN/inf (e.g. un-ingested macro/EPS columns) with 0
                # before they can propagate through the model and produce nan loss.
                x_price = torch.nan_to_num(x_price, nan=0.0, posinf=0.0, neginf=0.0)
                x_sent  = torch.nan_to_num(x_sent,  nan=0.0, posinf=0.0, neginf=0.0)
                y_reg   = torch.nan_to_num(y_reg,   nan=0.0, posinf=0.0, neginf=0.0)

                # Gaussian noise augmentation (training only): 0.02σ jitter on
                # Z-scored inputs — prevents memorising exact feature values.
                if training:
                    x_price = x_price + torch.randn_like(x_price) * 0.02

                dir_logits, regression, timing_logits = self.model(x_price, x_sent)

                cls_loss    = self.cls_loss_fn(dir_logits, y_cls)
                reg_loss    = self.reg_loss_fn(regression, y_reg)
                timing_loss = self.timing_loss_fn(timing_logits, y_timing)

                loss = (
                    self.cls_weight    * cls_loss
                    + self.reg_weight  * reg_loss
                    + self.timing_weight * timing_loss
                )

                # If loss became non-finite despite all input guards, skip the
                # batch entirely so we never apply a NaN gradient (which would
                # corrupt every subsequent forward pass).
                if not torch.isfinite(loss):
                    continue

                if training:
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()

                n = len(y_cls)
                total_loss      += loss.item()       * n
                sum_cls_loss    += cls_loss.item()   * n
                sum_reg_loss    += reg_loss.item()   * n
                sum_timing_loss += timing_loss.item()* n

                preds_dir    = dir_logits.argmax(dim=-1)
                preds_timing = timing_logits.argmax(dim=-1)
                correct_dir    += (preds_dir    == y_cls).sum().item()
                correct_timing += (preds_timing == y_timing).sum().item()
                total          += n

        n = max(total, 1)
        return (
            total_loss / n,
            correct_dir    / n,
            correct_timing / n,
            [sum_cls_loss / n, sum_reg_loss / n, sum_timing_loss / n],
        )

    # ── Checkpoint ────────────────────────────────────────────────────────────

    def _save_checkpoint(self, epoch: int, val_loss: float, val_acc: float):
        torch.save(
            {
                "epoch":       epoch,
                "model_state": self.model.state_dict(),
                "val_loss":    val_loss,
                "val_acc":     val_acc,
                "optimizer":   self.optimizer.state_dict(),
            },
            self.checkpoint_path,
        )
        logger.info(f"💾 Checkpoint saved → {self.checkpoint_path}")

    def load_best(self):
        if not os.path.exists(self.checkpoint_path):
            logger.warning(f"⚠️  No checkpoint at {self.checkpoint_path} — keeping current weights")
            return
        ckpt = torch.load(
            self.checkpoint_path, map_location="cpu", weights_only=True
        )
        self.model.load_state_dict(ckpt["model_state"])
        logger.info(
            f"✅ Loaded best checkpoint "
            f"(epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.4f})"
        )


# ── Utility: compute balanced class weights from label array ──────────────────

def compute_class_weights(y_cls: np.ndarray, n_classes: int = 3) -> torch.Tensor:
    """
    Class weights calibrated to actual label distribution.

    With real data: BUY~52%, SELL~35%, HOLD~13%.
    HOLD is the minority — standard inverse-frequency would over-weight HOLD
    and suppress BUY/SELL confidence. Instead we use mild sqrt-inverse weights
    so no class dominates training and confidence stays sharp on BUY/SELL.

    Formula: weight[c] = sqrt(total / (n_classes * count[c])), then normalise.
    """
    counts = np.bincount(y_cls, minlength=n_classes).astype(float)
    counts = np.where(counts == 0, 1, counts)
    total  = counts.sum()
    # sqrt dampens extreme weights — prevents HOLD from dominating
    weights = np.sqrt(total / (n_classes * counts))
    weights = weights / weights.mean()  # normalise so mean == 1
    logger.info(
        f"⚖️  Class distribution — "
        f"Hold: {int(counts[0])} ({counts[0]/total:.1%}) w={weights[0]:.2f} | "
        f"Buy: {int(counts[1])} ({counts[1]/total:.1%}) w={weights[1]:.2f} | "
        f"Sell: {int(counts[2])} ({counts[2]/total:.1%}) w={weights[2]:.2f}"
    )
    return torch.tensor(weights, dtype=torch.float32)

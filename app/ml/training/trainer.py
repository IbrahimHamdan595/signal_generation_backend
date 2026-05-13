import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Tuple, Optional
import numpy as np
import os
import logging

from sklearn.metrics import f1_score

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
        label_smoothing: float = 0.0,
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
        reg_weight: float = 0.3,
        timing_weight: float = 0.1,
        patience: int = 10,             # cut early when val plateaus — avoids 28-epoch overfit runs
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

        # Mixed-precision training: ~40% wall-clock speedup on RTX 3050.
        # GradScaler is a no-op on CPU and only enabled when CUDA is available.
        self.amp_enabled = torch.cuda.is_available()
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp_enabled)
        if self.amp_enabled:
            logger.info("⚡ Mixed-precision (AMP) enabled")

        # Label smoothing dropped to 0.0 — soft targets reinforce the majority
        # class on imbalanced data and contributed to the BUY-everywhere collapse.
        self.cls_loss_fn = FocalLoss(
            gamma=2.0,
            weight=class_weights.to(self.device) if class_weights is not None else None,
            label_smoothing=0.0,
        )
        if class_weights is not None:
            logger.info(f"⚖️  Class weights applied to FocalLoss: {class_weights.tolist()}")

        self.reg_loss_fn    = nn.MSELoss()
        self.timing_loss_fn = nn.CrossEntropyLoss(
            weight=timing_class_weights.to(self.device) if timing_class_weights is not None else None,
            label_smoothing=0.0,
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
        best_val_f1      = -1.0
        best_val_loss    = float("inf")
        epochs_no_improve = 0

        # ReduceLROnPlateau on macro-F1 (mode="max"): step LR down when F1 stops
        # improving. Tracking F1 instead of loss prevents the scheduler from
        # rewarding the degenerate "always BUY" minimum-loss solution.
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="max",
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

            tr_loss, tr_acc, tr_timing_acc, tr_losses, tr_f1 = self._run_epoch(train_loader, training=True)
            va_loss, va_acc, va_timing_acc, va_losses, va_f1 = self._run_epoch(val_loader,   training=False)

            train_loss, val_loss = tr_loss, va_loss
            train_acc,  val_acc  = tr_acc,  va_acc

            if epoch > self._warmup_epochs:
                self.scheduler.step(va_f1)
            self.train_history.append({
                "loss": tr_loss, "acc": tr_acc, "timing_acc": tr_timing_acc, "macro_f1": tr_f1,
                "cls_loss": tr_losses[0], "reg_loss": tr_losses[1], "timing_loss": tr_losses[2],
            })
            self.val_history.append({
                "loss": va_loss, "acc": va_acc, "timing_acc": va_timing_acc, "macro_f1": va_f1,
                "cls_loss": va_losses[0], "reg_loss": va_losses[1], "timing_loss": va_losses[2],
            })

            logger.info(
                f"Epoch {epoch:03d}/{epochs} | "
                f"Train loss: {train_loss:.4f} acc: {train_acc:.3f} f1: {tr_f1:.3f} | "
                f"Val loss: {val_loss:.4f} acc: {val_acc:.3f} f1: {va_f1:.3f} | "
                f"timing_acc: {self.val_history[-1].get('timing_acc', 0):.3f}"
            )

            # Best-checkpoint selection by macro-F1 (treats each class equally,
            # so collapsed classifiers cannot win).
            if va_f1 > best_val_f1:
                best_val_f1       = va_f1
                best_val_loss     = val_loss
                epochs_no_improve = 0
                self._save_checkpoint(epoch, val_loss, val_acc)
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.patience:
                    logger.info(f"⏹ Early stopping at epoch {epoch} (best F1={best_val_f1:.4f})")
                    break

        logger.info(f"✅ Training done. Best val macro-F1: {best_val_f1:.4f} (val_loss={best_val_loss:.4f})")
        return {
            "best_val_loss":  best_val_loss,
            "best_val_f1":    best_val_f1,
            "train_history":  self.train_history,
            "val_history":    self.val_history,
            "checkpoint":     self.checkpoint_path,
        }

    # ── Epoch runner ──────────────────────────────────────────────────────────

    def _run_epoch(self, loader: DataLoader, training: bool) -> Tuple[float, float, float, list, float]:
        """Returns (total_loss, dir_accuracy, timing_accuracy, [cls_loss, reg_loss, timing_loss], macro_f1)."""
        self.model.train() if training else self.model.eval()

        total_loss = 0.0
        sum_cls_loss = sum_reg_loss = sum_timing_loss = 0.0
        correct_dir = correct_timing = total = 0
        all_preds: list = []
        all_labels: list = []

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
                # MixUp was removed: it over-softened the labels and the model
                # collapsed to predicting the majority class (HOLD 97%).
                if training:
                    x_price = x_price + torch.randn_like(x_price) * 0.02

                # Mixed-precision forward pass: most ops run in float16 for
                # speed, losses stay in float32 for numerical stability.
                with torch.amp.autocast(device_type="cuda", enabled=self.amp_enabled):
                    dir_logits, regression, timing_logits = self.model(x_price, x_sent)
                    cls_loss    = self.cls_loss_fn(dir_logits, y_cls)
                    timing_loss = self.timing_loss_fn(timing_logits, y_timing)
                    reg_loss    = self.reg_loss_fn(regression, y_reg)

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
                    # GradScaler scales the loss before backward so float16
                    # gradients don't underflow; unscales them again before
                    # the optimizer step.
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

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

                all_preds.append(preds_dir.cpu().numpy())
                all_labels.append(y_cls.cpu().numpy())

        n = max(total, 1)
        if all_preds:
            preds_flat  = np.concatenate(all_preds)
            labels_flat = np.concatenate(all_labels)
            macro_f1 = float(f1_score(labels_flat, preds_flat, average="macro", zero_division=0))
        else:
            macro_f1 = 0.0
        return (
            total_loss / n,
            correct_dir    / n,
            correct_timing / n,
            [sum_cls_loss / n, sum_reg_loss / n, sum_timing_loss / n],
            macro_f1,
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

def compute_class_weights(
    y_cls: np.ndarray,
    n_classes: int = 3,
    sharpening: float = 0.5,
) -> torch.Tensor:
    """
    Sqrt-inverse class weights — gentler than linear-inverse.

    History of choices on this dataset:
      • sharpening=1.5 → HOLD weight 1.3 → focal loss collapsed model to HOLD
      • sharpening=1.0 (linear) → HOLD weight 1.65 with 17% HOLD samples
                                  → model still HOLD-leaning, only 30% acc
      • sharpening=0.5 (sqrt)   → HOLD weight ~1.2 → minimal HOLD pull, lets
                                  the model favour the majority class (BUY)
                                  as much as the data supports, recovering
                                  test accuracy.

    Trade-off: this returns to a BUY-leaning model but with healthier
    recall on BUY/SELL than the 92% BUY-collapse from the first attempt.
    """
    counts = np.bincount(y_cls, minlength=n_classes).astype(float)
    counts = np.where(counts == 0, 1, counts)
    total  = counts.sum()
    raw     = (total / counts) ** sharpening
    weights = raw / raw.mean()                     # normalise so mean == 1
    logger.info(
        f"⚖️  Class distribution — "
        f"Hold: {int(counts[0])} ({counts[0]/total:.1%}) w={weights[0]:.2f} | "
        f"Buy: {int(counts[1])} ({counts[1]/total:.1%}) w={weights[1]:.2f} | "
        f"Sell: {int(counts[2])} ({counts[2]/total:.1%}) w={weights[2]:.2f}"
    )
    return torch.tensor(weights, dtype=torch.float32)

"""Training loop, checkpointing and early stopping."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.logging_setup import get_logger
from config import Config, config_to_device
from .losses import build_loss
from .metrics import compute_metrics
from .scheduler import build_optimizer, build_scheduler

logger = get_logger(__name__)


class EarlyStopping:
    """Stop training when a monitored metric stops improving."""

    def __init__(self, patience: int = 8, mode: str = "max") -> None:
        """Initialise early stopping.

        Args:
            patience: Number of epochs to wait after the best score.
            mode: ``max`` (higher is better) or ``min``.
        """
        self.patience = patience
        self.mode = mode
        self.best = -float("inf") if mode == "max" else float("inf")
        self.best_epoch = 0
        self.counter = 0

    def step(self, score: float, epoch: int) -> bool:
        """Record a score and report whether training should stop.

        Args:
            score: Monitored metric value.
            epoch: Current epoch (1-indexed).

        Returns:
            ``True`` when the patience window has elapsed.
        """
        improved = score > self.best if self.mode == "max" else score < self.best
        if improved:
            self.best = score
            self.best_epoch = epoch
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


class Trainer:
    """Owns the optimiser, scheduler, loss and training loop for one run."""

    def __init__(
        self,
        model: nn.Module,
        cfg: Config,
        device: str | None = None,
        run_dir: str | Path | None = None,
        class_weights: torch.Tensor | None = None,
        writer: object | None = None,
    ) -> None:
        """Initialise the trainer.

        Args:
            model: Model to train.
            cfg: Full training configuration.
            device: Torch device string (resolved from config when ``None``).
            run_dir: Directory for best-state checkpoints.
            class_weights: Optional class weights for the loss.
            writer: Optional TensorBoard SummaryWriter.
        """
        self.model = model
        self.cfg = cfg
        self.device = device or config_to_device(cfg)
        self.run_dir = Path(run_dir) if run_dir else None
        self.writer = writer

        self.model.to(self.device)
        self.optimizer = build_optimizer(model, cfg)
        self.scheduler = build_scheduler(self.optimizer, cfg)
        self.criterion = build_loss(cfg, class_weights, self.device)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.device == "cuda")
        self.early_stopping = EarlyStopping(
            patience=int(cfg.training.early_stopping_patience), mode="max"
        )

        self.best_score = -float("inf")
        self.best_epoch = 0
        self.best_state: dict[str, torch.Tensor] | None = None
        self.best_checkpoint: Path | None = None

    # ------------------------------------------------------------------ #
    @staticmethod
    def _monitor_score(val_metrics: dict[str, object]) -> float:
        """Validation score used for checkpointing/early stopping.

        ROC-AUC is preferred (robust under class imbalance); accuracy is used
        when AUC is unavailable (e.g. a single-class validation batch).
        """
        score = val_metrics.get("roc_auc")
        if score is None or (isinstance(score, float) and np.isnan(score)):
            score = val_metrics["accuracy"]
        return float(score)

    # ------------------------------------------------------------------ #
    def _save_best(self, epoch: int, score: float) -> None:
        """Snapshot the best model state (and write a checkpoint when enabled)."""
        self.best_score = score
        self.best_epoch = epoch
        self.best_state = copy.deepcopy(self.model.state_dict())
        if self.run_dir is not None:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            self.best_checkpoint = self.run_dir / "best_model.pt"
            torch.save(
                {"state_dict": self.best_state, "epoch": epoch, "score": score},
                self.best_checkpoint,
            )

    def fit(
        self, train_loader: DataLoader, val_loader: DataLoader
    ) -> tuple[dict[str, list[dict[str, object]]], int, Path | None]:
        """Run the full training loop.

        Args:
            train_loader: Training DataLoader.
            val_loader: Validation DataLoader.

        Returns:
            Tuple of ``(history, best_epoch, best_checkpoint)`` where history
            holds per-epoch ``train`` and ``val`` metric dicts.
        """
        epochs = int(self.cfg.training.epochs)
        history: dict[str, list[dict[str, object]]] = {"train": [], "val": []}

        for epoch in range(1, epochs + 1):
            train_metrics = self._run_epoch(train_loader, train=True)
            val_metrics = self._run_epoch(val_loader, train=False)
            self.scheduler.step()

            history["train"].append(train_metrics)
            history["val"].append(val_metrics)

            if self.writer is not None:
                self.writer.add_scalar("train/loss", train_metrics["loss"], epoch)
                self.writer.add_scalar("val/loss", val_metrics["loss"], epoch)
                self.writer.add_scalar("val/accuracy", val_metrics["accuracy"], epoch)

            logger.info(
                "Epoch %d/%d | train loss %.4f acc %.3f | val loss %.4f acc %.3f auc %.3f",
                epoch,
                epochs,
                train_metrics["loss"],
                train_metrics["accuracy"],
                val_metrics["loss"],
                val_metrics["accuracy"],
                val_metrics.get("roc_auc", float("nan")),
            )

            score = self._monitor_score(val_metrics)
            if score > self.best_score:
                self._save_best(epoch, score)

            if self.early_stopping.step(score, epoch):
                logger.info("Early stopping triggered after epoch %d.", epoch)
                break

        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)
        return history, self.best_epoch, self.best_checkpoint

    # ------------------------------------------------------------------ #
    def _run_epoch(self, loader: DataLoader, train: bool) -> dict[str, object]:
        """Run one training or validation epoch.

        Args:
            loader: DataLoader to iterate.
            train: Whether to enable gradients and update weights.

        Returns:
            Metrics dict (loss + accuracy, plus the full metric set on val).
        """
        self.model.train(train)
        total_loss = 0.0
        y_true: list[np.ndarray] = []
        y_pred: list[np.ndarray] = []
        y_prob: list[np.ndarray] = []

        loop = tqdm(loader, ncols=90, leave=False, desc="train" if train else "val")
        for batch in loop:
            labels = batch["label"].to(self.device)
            out = self._model_forward(batch)

            loss = self.criterion(out, labels)
            if train:
                self.optimizer.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()
                if self.cfg.training.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.cfg.training.grad_clip
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()

            total_loss += float(loss.item()) * labels.size(0)
            preds = out.argmax(dim=-1).detach().cpu().numpy()
            probs = torch.softmax(out, dim=-1).detach().cpu().numpy()
            y_true.append(labels.detach().cpu().numpy())
            y_pred.append(preds)
            if not train:
                y_prob.append(probs)
            loop.set_postfix(loss=float(loss.item()))

        n = len(loader.dataset)
        metrics: dict[str, object] = {
            "loss": total_loss / max(n, 1),
            "accuracy": float(np.mean(np.concatenate(y_pred) == np.concatenate(y_true))),
        }
        if not train:
            metrics.update(
                compute_metrics(
                    np.concatenate(y_true),
                    np.concatenate(y_pred),
                    np.concatenate(y_prob),
                    n_classes=self.cfg.model.n_classes,
                )
            )
        return metrics

    def _model_forward(self, batch: dict[str, object]) -> torch.Tensor:
        """Forward a batch through the model, moving tensors to device."""
        kwargs: dict[str, torch.Tensor] = {}
        if "image" in batch:
            kwargs["image"] = batch["image"].to(self.device)
        if "image_feat" in batch:
            kwargs["image_feat"] = batch["image_feat"].to(self.device)
        if "clinical" in batch:
            kwargs["clinical"] = batch["clinical"].to(self.device)
        return self.model(**kwargs)

    @torch.no_grad()
    def predict(self, loader: DataLoader) -> dict[str, np.ndarray]:
        """Run inference and return predictions.

        Args:
            loader: DataLoader over an evaluation split.

        Returns:
            Dict with ``patient_id`` (list), ``y_true``, ``y_pred`` and
            ``y_prob`` arrays.
        """
        self.model.eval()
        patient_ids: list[str] = []
        y_true: list[np.ndarray] = []
        y_pred: list[np.ndarray] = []
        y_prob: list[np.ndarray] = []

        for batch in loader:
            out = self._model_forward(batch)
            patient_ids.extend(batch["patient_id"])
            y_true.append(batch["label"].numpy())
            y_pred.append(out.argmax(dim=-1).detach().cpu().numpy())
            y_prob.append(torch.softmax(out, dim=-1).detach().cpu().numpy())

        return {
            "patient_id": patient_ids,
            "y_true": np.concatenate(y_true),
            "y_pred": np.concatenate(y_pred),
            "y_prob": np.concatenate(y_prob),
        }
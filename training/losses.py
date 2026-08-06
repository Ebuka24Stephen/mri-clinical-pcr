"""Loss construction and class weighting."""

from __future__ import annotations

import pandas as pd
import torch
import torch.nn as nn

from config import Config


def compute_class_weights(
    manifest: pd.DataFrame, n_classes: int, mode: str = "balanced"
) -> torch.Tensor | None:
    """Compute per-class weights for the loss.

    Args:
        manifest: Manifest with a ``label`` column.
        n_classes: Number of classes.
        mode: ``balanced`` for inverse-frequency weights or ``none``.

    Returns:
        Class weight tensor, or ``None`` when ``mode`` is ``none``.
    """
    if mode != "balanced":
        return None
    counts = manifest["label"].value_counts()
    weights = torch.zeros(n_classes, dtype=torch.float32)
    for cls in range(n_classes):
        n = float(counts.get(cls, 0))
        weights[cls] = len(manifest) / (n_classes * max(n, 1.0))
    return weights


def build_loss(
    cfg: Config,
    class_weights: torch.Tensor | None = None,
    device: str = "cpu",
) -> nn.Module:
    """Build the cross-entropy loss according to configuration.

    Args:
        cfg: Training configuration (class weighting, label smoothing).
        class_weights: Optional precomputed class weights.
        device: Target device for the weight tensor.

    Returns:
        An ``nn.CrossEntropyLoss`` instance.
    """
    if class_weights is not None:
        class_weights = class_weights.to(device)
    return nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=float(cfg.training.label_smoothing),
    )
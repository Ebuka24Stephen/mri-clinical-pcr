"""Optimizer and scheduler construction."""

from __future__ import annotations

import torch.optim as optim

from config import Config


def build_optimizer(model: object, cfg: Config) -> optim.Optimizer:
    """Build the optimizer selected by the configuration.

    Args:
        model: Module whose trainable parameters should be optimised.
        cfg: Training configuration.

    Returns:
        A configured optimizer instance.

    Raises:
        ValueError: For unknown optimizer names.
    """
    params = [p for p in model.parameters() if p.requires_grad]
    lr = float(cfg.training.lr)
    wd = float(cfg.training.weight_decay)
    name = cfg.training.optimizer.lower()
    if name == "adam":
        return optim.Adam(params, lr=lr, weight_decay=wd)
    if name == "adamw":
        return optim.AdamW(params, lr=lr, weight_decay=wd)
    if name == "sgd":
        return optim.SGD(params, lr=lr, weight_decay=wd, momentum=0.9)
    raise ValueError(f"Unknown optimizer: {cfg.training.optimizer}")


def build_scheduler(optimizer: optim.Optimizer, cfg: Config) -> object:
    """Build the LR scheduler selected by the configuration.

    Args:
        optimizer: The optimiser to schedule.
        cfg: Training configuration.

    Returns:
        A scheduler instance.
    """
    name = cfg.training.scheduler.lower()
    if name == "none":
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
    if name == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(cfg.training.epochs, 1)
        )
    if name == "step":
        return optim.lr_scheduler.StepLR(
            optimizer, step_size=max(cfg.training.epochs // 3, 1), gamma=0.5
        )
    raise ValueError(f"Unknown scheduler: {cfg.training.scheduler}")
"""Reproducibility helpers: seeding and dataloader worker seeding."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch (CPU + CUDA) RNGs.

    Args:
        seed: Integer seed to use for all RNGs.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Keep cuDNN deterministic for reproducible runs.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def worker_init_fn(worker_id: int) -> None:
    """Seed each DataLoader worker with a deterministic, worker-unique seed.

    Args:
        worker_id: Index of the worker process.
    """
    np.random.seed(np.random.randint(0, 2**31) + worker_id)
    random.seed(worker_id)

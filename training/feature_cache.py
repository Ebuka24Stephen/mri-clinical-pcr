"""Frozen-backbone image embedding cache.

On a CPU-only machine, re-running a ResNet50 over every MRI slice each epoch is
wasteful. This module precomputes image embeddings once (under ``no_grad``,
backbone in eval mode) and caches them to disk as ``{image_path: embedding}``
maps reused across training and evaluation runs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from utils.logging_setup import get_logger

logger = get_logger(__name__)


class FeatureCache:
    """Disk-backed cache of image embeddings keyed by image path."""

    def __init__(self, cache_dir: str | Path) -> None:
        """Initialise the cache.

        Args:
            cache_dir: Directory where cached embedding files are stored.
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def path(self, tag: str) -> Path:
        """Return the file path for a named cache.

        Args:
            tag: Cache name (e.g. ``resnet50_224``).

        Returns:
            The ``.npz`` path for the cache.
        """
        return self.cache_dir / f"{tag}.npz"

    def exists(self, tag: str) -> bool:
        """Return whether a named cache already exists.

        Args:
            tag: Cache name.

        Returns:
            ``True`` when the cache file exists.
        """
        return self.path(tag).exists()

    def save(self, tag: str, mapping: dict[str, np.ndarray]) -> None:
        """Persist an ``image_path -> embedding`` mapping.

        Args:
            tag: Cache name.
            mapping: Mapping of image path strings to 1-D float arrays.
        """
        keys = np.array(list(mapping.keys()))
        values = np.stack([np.asarray(v, dtype=np.float32) for v in mapping.values()])
        np.savez(self.path(tag), keys=keys, values=values)
        logger.info("Saved feature cache '%s' with %d entries.", tag, len(mapping))

    def load(self, tag: str) -> dict[str, np.ndarray] | None:
        """Load a cached mapping if present.

        Args:
            tag: Cache name.

        Returns:
            The mapping, or ``None`` if the cache does not exist.
        """
        path = self.path(tag)
        if not path.exists():
            return None
        with np.load(path, allow_pickle=False) as data:
            keys = [str(k) for k in data["keys"]]
            values = data["values"]
        logger.info("Loaded feature cache '%s' with %d entries.", tag, len(keys))
        return {k: values[i] for i, k in enumerate(keys)}

    @staticmethod
    def compute(
        encoder: nn.Module,
        dataset: Dataset,
        batch_size: int,
        device: str,
        num_workers: int = 0,
    ) -> dict[str, np.ndarray]:
        """Compute embeddings for every sample of a raw-image dataset.

        The encoder is switched to eval mode and gradients are disabled. Returns
        a mapping keyed by the sample ``image_path``.

        Args:
            encoder: Image encoder module.
            dataset: Dataset whose samples contain ``image`` and ``image_path``.
            batch_size: Batch size for the inference loop.
            device: Torch device string.
            num_workers: DataLoader workers.

        Returns:
            Mapping ``image_path -> embedding`` (float32).
        """
        encoder.to(device).eval()
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        mapping: dict[str, np.ndarray] = {}
        with torch.no_grad():
            for batch in tqdm(loader, desc="Caching image features", ncols=80):
                images = batch["image"].to(device)
                paths = batch["image_path"]
                feats = encoder(images).detach().cpu().numpy()
                for p, f in zip(paths, feats):
                    mapping[str(p)] = np.asarray(f, dtype=np.float32)
        return mapping
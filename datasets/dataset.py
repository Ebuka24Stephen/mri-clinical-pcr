"""PyTorch datasets and collation for Cancer-Net BCa.

A single :class:`CancerNetDataset` serves both the unimodal (MRI-only) and
multimodal (MRI + clinical) models. Each sample is a dict with keys chosen from
``{patient_id, image_path, image, image_feat, clinical, label}`` depending on
the enabled modalities and whether frozen-backbone image features are cached.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class CancerNetDataset(Dataset):
    """Patient/slice-level dataset returning modality dicts.

    Args:
        rows: Manifest slice (one row per ``(patient_id, image_path)``).
        clinical_map: Optional mapping ``patient_id -> clinical vector``.
        image_cache: Optional mapping ``image_path -> cached embedding``.
        transform: Image transform (ignored when ``image_cache`` is used).
        use_image: Whether to load/produce image inputs.
        use_clinical: Whether to return clinical features.
    """

    def __init__(
        self,
        rows: pd.DataFrame,
        clinical_map: dict[str, np.ndarray] | None = None,
        image_cache: dict[str, np.ndarray] | None = None,
        transform: transforms.Compose | None = None,
        use_image: bool = True,
        use_clinical: bool = True,
    ) -> None:
        self.rows = rows.reset_index(drop=True)
        self.clinical_map = clinical_map or {}
        self.image_cache = image_cache or {}
        self.transform = transform
        self.use_image = use_image
        self.use_clinical = use_clinical

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str | int]:
        row = self.rows.iloc[idx]
        out: dict[str, torch.Tensor | str | int] = {
            "patient_id": str(row["patient_id"]),
            "image_path": str(row["image_path"]),
            "label": int(row["label"]),
        }

        if self.use_image:
            image_path = str(row["image_path"])
            if image_path in self.image_cache:
                out["image_feat"] = torch.from_numpy(
                    np.asarray(self.image_cache[image_path], dtype=np.float32)
                )
            else:
                if self.transform is None:
                    raise ValueError("Raw images require an image transform.")
                out["image"] = self._load_image(image_path, self.transform)

        if self.use_clinical:
            out["clinical"] = torch.from_numpy(
                np.asarray(self.clinical_map[str(row["patient_id"])], dtype=np.float32)
            )

        return out

    @staticmethod
    def _load_image(path: str, transform: transforms.Compose) -> torch.Tensor:
        """Load an MRI slice PNG and apply a transform.

        Args:
            path: Image file path.
            transform: Transform composition.

        Returns:
            Image tensor of shape ``(C, H, W)``.
        """
        img = Image.open(path).convert("RGB")
        return transform(img)


def collate_batch(batch: list[dict[str, torch.Tensor | str | int]]) -> dict[str, object]:
    """Collate a list of modality dicts into a stacked batch dict.

    Args:
        batch: List of samples produced by :class:`CancerNetDataset`.

    Returns:
        Batch dict with ``patient_id`` (list[str]) and stacked tensors for the
        keys present in every sample.
    """
    out: dict[str, object] = {"patient_id": [str(b["patient_id"]) for b in batch]}
    keys = set(batch[0].keys())
    for key in keys - {"patient_id"}:
        if all(key in b for b in batch):
            tensors = [b[key] for b in batch]
            if isinstance(tensors[0], torch.Tensor):
                out[key] = torch.stack(tensors)
            else:
                out[key] = tensors
    out["label"] = torch.tensor([int(b["label"]) for b in batch], dtype=torch.long)
    return out
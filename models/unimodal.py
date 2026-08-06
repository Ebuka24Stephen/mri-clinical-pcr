"""Unimodal baseline: MRI-only pCR prediction.

Pipeline: MRI slice -> ResNet50 -> image embedding -> MLP head -> pCR logits.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from utils.logging_setup import get_logger
from .components import MLPHead
from .image_encoder import ImageEncoder

logger = get_logger(__name__)


class UnimodalModel(nn.Module):
    """Image-embedding -> MLP classification head (binary pCR)."""

    def __init__(
        self,
        image_embedding_dim: int = 512,
        hidden: list[int] | None = None,
        n_classes: int = 2,
        dropout: float = 0.3,
        freeze_backbone: bool = True,
        backbone: str = "resnet50",
        weights: str = "IMAGENET1K_V1",
    ) -> None:
        """Initialise the unimodal model.

        Args:
            image_embedding_dim: Dimensionality of image embeddings.
            hidden: Hidden layer widths of the classification head.
            n_classes: Number of output classes.
            dropout: Dropout probability in the head.
            freeze_backbone: Freeze the backbone (use with cached features).
            backbone: Torchvision backbone name.
            weights: Torchvision weights enum name.
        """
        super().__init__()
        hidden = hidden or [128]
        self.image_encoder = ImageEncoder(
            backbone=backbone,
            weights=weights,
            out_dim=image_embedding_dim,
            freeze=freeze_backbone,
        )
        self.head = MLPHead(image_embedding_dim, hidden, n_classes, dropout)
        self.n_classes = n_classes
        logger.info(
            "UnimodalModel built: image_dim=%d hidden=%s classes=%d",
            image_embedding_dim,
            hidden,
            n_classes,
        )

    def forward(
        self,
        image: torch.Tensor | None = None,
        image_feat: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Classify an MRI batch.

        Args:
            image: Raw image tensor ``(B, 3, H, W)``; encoded by the backbone.
            image_feat: Precomputed image embeddings ``(B, image_dim)``.

        Returns:
            Class logits ``(B, n_classes)``.

        Raises:
            ValueError: If neither ``image`` nor ``image_feat`` is provided.
        """
        if image_feat is not None:
            feat = image_feat
        elif image is not None:
            feat = self.image_encoder(image)
        else:
            raise ValueError("UnimodalModel.forward requires image or image_feat.")
        return self.head(feat)
"""Image encoders built on ImageNet-pretrained backbones.

The :class:`ImageEncoder` replaces the classifier head with a trainable linear
projection producing the configurable ``image_embedding_dim``-dimensional
embedding. When ``freeze_backbone=True`` the backbone weights are frozen so a
CPU workflow can precompute image embeddings once and reuse them.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as models

from utils.logging_setup import get_logger

logger = get_logger(__name__)


class ImageEncoder(nn.Module):
    """Pretrained CNN backbone followed by a linear embedding head."""

    def __init__(
        self,
        backbone: str = "resnet50",
        weights: str = "IMAGENET1K_V1",
        out_dim: int = 512,
        freeze: bool = True,
    ) -> None:
        """Initialise the encoder.

        Args:
            backbone: Torchvision model name (e.g. ``resnet50``).
            weights: Torchvision weights enum name or ``None``.
            out_dim: Dimensionality of the output embedding.
            freeze: Whether to freeze all backbone parameters.
        """
        super().__init__()
        net = models.get_model(backbone, weights=weights or None)
        in_features = getattr(net.fc, "in_features")
        net.fc = nn.Linear(in_features, out_dim)
        self.backbone = net
        self.out_dim = out_dim
        self.freeze = freeze

        if freeze:
            for param in net.parameters():
                param.requires_grad_(False)
            net.fc.requires_grad_(True)
        logger.info(
            "ImageEncoder(%s) built: out_dim=%d, freeze=%s",
            backbone,
            out_dim,
            freeze,
        )

    @property
    def feature_dim(self) -> int:
        """Dimensionality of the produced embedding."""
        return self.out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Embed a batch of images.

        Args:
            x: Image tensor of shape ``(B, 3, H, W)``.

        Returns:
            Embedding tensor of shape ``(B, out_dim)``.
        """
        return self.backbone(x)
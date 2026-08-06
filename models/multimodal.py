"""Multimodal MRI + clinical model with late feature fusion.

Pipeline:
    MRI slice -> ResNet50 -> image embedding
    clinical data -> MLP -> clinical embedding
    concatenate embeddings -> fully connected head -> pCR logits

Fusion is a simple concatenation (late fusion) as required; no attention or
other complex fusion strategies are used.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from config import ModelConfig
from utils.logging_setup import get_logger
from .clinical_mlp import ClinicalMLP
from .components import MLPHead
from .image_encoder import ImageEncoder

logger = get_logger(__name__)


class MultimodalModel(nn.Module):
    """Late-fusion (concatenation) MRI + clinical classifier."""

    def __init__(
        self,
        n_classes: int,
        clinical_dim: int,
        image_dim: int = 512,
        cfg: ModelConfig | None = None,
        freeze_backbone: bool = True,
        backbone: str = "resnet50",
        weights: str = "IMAGENET1K_V1",
    ) -> None:
        """Initialise the multimodal model.

        Args:
            n_classes: Number of output classes.
            clinical_dim: Raw clinical feature dimensionality.
            image_dim: Image embedding dimensionality.
            cfg: Model configuration (hidden sizes, embeddings, dropout).
            freeze_backbone: Freeze the image backbone (cached-feature mode).
            backbone: Torchvision backbone name.
            weights: Torchvision weights enum name.
        """
        super().__init__()
        cfg = cfg or ModelConfig()
        self.n_classes = n_classes

        self.image_encoder = ImageEncoder(
            backbone=backbone,
            weights=weights,
            out_dim=image_dim,
            freeze=freeze_backbone,
        )
        self.clinical_mlp = ClinicalMLP(
            in_dim=clinical_dim,
            hidden_dim=cfg.fusion_hidden[0] if cfg.fusion_hidden else 64,
            embedding_dim=cfg.clinical_embedding_dim,
            dropout=cfg.dropout,
        )
        head_in = image_dim + cfg.clinical_embedding_dim
        self.head = MLPHead(head_in, cfg.fusion_hidden, n_classes, cfg.dropout)
        logger.info(
            "MultimodalModel built: image_dim=%d clinical_dim=%d head_in=%d classes=%d",
            image_dim,
            clinical_dim,
            head_in,
            n_classes,
        )

    def forward(
        self,
        image: torch.Tensor | None = None,
        image_feat: torch.Tensor | None = None,
        clinical: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Fuse MRI and clinical inputs and classify.

        Args:
            image: Raw image batch ``(B, 3, H, W)`` (encoded internally).
            image_feat: Precomputed image embeddings ``(B, image_dim)``.
            clinical: Clinical feature batch ``(B, clinical_dim)``.

        Returns:
            Class logits ``(B, n_classes)``.

        Raises:
            ValueError: If image or clinical inputs are missing.
        """
        if clinical is None:
            raise ValueError("MultimodalModel requires clinical input.")
        if image_feat is not None:
            image_emb = image_feat
        elif image is not None:
            image_emb = self.image_encoder(image)
        else:
            raise ValueError("MultimodalModel requires image or image_feat.")

        clinical_emb = self.clinical_mlp(clinical)
        fused = torch.cat([image_emb, clinical_emb], dim=-1)
        return self.head(fused)
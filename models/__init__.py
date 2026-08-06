"""Model definitions and factory."""

from __future__ import annotations

from config import Config
from .clinical_mlp import ClinicalMLP
from .components import MLPHead
from .image_encoder import ImageEncoder
from .multimodal import MultimodalModel
from .unimodal import UnimodalModel


def build_model(
    cfg: Config,
    clinical_dim: int = 0,
) -> UnimodalModel | MultimodalModel:
    """Construct the model selected by the configuration.

    Args:
        cfg: Full experiment configuration.
        clinical_dim: Raw clinical feature dimensionality (0 = image-only).

    Returns:
        A unimodal (MRI-only) or multimodal model.

    Raises:
        ValueError: If the config requests an invalid modality combination.
    """
    if cfg.model.n_classes != 2:
        raise ValueError("Cancer-Net BCa is a binary pCR task; n_classes must be 2.")
    if clinical_dim <= 0:
        return UnimodalModel(
            image_embedding_dim=cfg.image.image_embedding_dim,
            hidden=cfg.model.fusion_hidden,
            n_classes=cfg.model.n_classes,
            dropout=cfg.model.dropout,
            freeze_backbone=cfg.image.freeze_backbone,
            backbone=cfg.image.backbone,
            weights=cfg.image.weights,
        )
    return MultimodalModel(
        n_classes=cfg.model.n_classes,
        clinical_dim=clinical_dim,
        image_dim=cfg.image.image_embedding_dim,
        cfg=cfg.model,
        freeze_backbone=cfg.image.freeze_backbone,
        backbone=cfg.image.backbone,
        weights=cfg.image.weights,
    )


__all__ = [
    "ClinicalMLP",
    "MLPHead",
    "ImageEncoder",
    "UnimodalModel",
    "MultimodalModel",
    "build_model",
]
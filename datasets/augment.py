"""Image transforms for training and validation.

Training augments with random horizontal flip, small rotations and colour
jitter as recommended for the MRI-only branch; validation applies resize and
normalisation only.
"""

from __future__ import annotations

import torch
from torchvision import transforms


def build_train_transforms(
    size: int = 224,
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> transforms.Compose:
    """Return the training image transforms (with augmentation).

    Args:
        size: Target spatial size.
        mean: Channel means for normalisation.
        std: Channel standard deviations for normalisation.

    Returns:
        A ``torchvision.transforms`` composition.
    """
    return transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def build_val_transforms(
    size: int = 224,
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> transforms.Compose:
    """Return the validation/test image transforms (no augmentation).

    Args:
        size: Target spatial size.
        mean: Channel means for normalisation.
        std: Channel standard deviations for normalisation.

    Returns:
        A ``torchvision.transforms`` composition.
    """
    return transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def build_transforms(
    train: bool,
    size: int = 224,
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> transforms.Compose:
    """Select train or validation transforms.

    Args:
        train: Whether to use the (augmented) training transforms.
        size: Target spatial size.
        mean: Channel means for normalisation.
        std: Channel standard deviations for normalisation.

    Returns:
        The chosen transform composition.
    """
    return (
        build_train_transforms(size, mean, std)
        if train
        else build_val_transforms(size, mean, std)
    )
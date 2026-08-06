"""Patient-level train/validation/test splitting.

Splits are computed on unique patient ids (never on individual image slices) so
the same patient cannot appear in more than one split, avoiding data leakage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from utils.logging_setup import get_logger

logger = get_logger(__name__)


def patient_stratified_split(
    manifest: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split patients into train/validation/test, stratified by label.

    Args:
        manifest: Patient-aligned manifest with ``patient_id`` and ``label``.
        train_ratio: Fraction of patients in the training split.
        val_ratio: Fraction of patients in the validation split (test is the
            remainder).
        seed: Random seed for reproducibility.

    Returns:
        Tuple of manifest row-index arrays ``(train_idx, val_idx, test_idx)``.

    Raises:
        ValueError: If the requested ratios leave no patients in a split.
    """
    patients = manifest.drop_duplicates("patient_id").reset_index(drop=True)
    X = np.arange(len(patients))
    y = patients["label"].values

    test_ratio = 1.0 - train_ratio - val_ratio
    if test_ratio <= 0:
        raise ValueError("train_ratio + val_ratio must be < 1.")

    train_val_idx, test_idx = next(
        StratifiedShuffleSplit(
            n_splits=1, test_size=test_ratio, random_state=seed
        ).split(X, y)
    )

    tv_patients = patients["patient_id"].iloc[train_val_idx].values
    tv_y = patients["label"].iloc[train_val_idx].values
    val_ratio_relative = val_ratio / (train_ratio + val_ratio)
    train_idx, val_idx = next(
        StratifiedShuffleSplit(
            n_splits=1, test_size=val_ratio_relative, random_state=seed
        ).split(train_val_idx, tv_y)
    )
    train_patients = tv_patients[train_idx]
    val_patients = tv_patients[val_idx]

    train_rows = manifest.index[manifest["patient_id"].isin(train_patients)].to_numpy()
    val_rows = manifest.index[manifest["patient_id"].isin(val_patients)].to_numpy()
    test_rows = manifest.index[manifest["patient_id"].isin(
        patients["patient_id"].iloc[test_idx].values
    )].to_numpy()

    logger.info(
        "Split: %d train / %d val / %d test patients (%d / %d / %d images)",
        len(train_patients),
        len(val_patients),
        len(test_idx),
        len(train_rows),
        len(val_rows),
        len(test_rows),
    )
    return train_rows, val_rows, test_rows
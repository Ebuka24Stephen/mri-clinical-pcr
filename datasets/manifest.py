"""Manifest construction: patient-aligned image slices, clinical features, label.

The manifest is a single :class:`pandas.DataFrame` that drives all downstream
dataset objects. One row exists per extracted MRI slice together with the
patient-level pCR label and the raw clinical feature columns. Building the
manifest here centralises label mapping and feature bookkeeping so the PyTorch
datasets stay simple and leak-free.
"""

from __future__ import annotations

import pandas as pd

from .loader import CancerNetBCaLoader

#: Standard label mapping for the binary pCR target.
PCR_LABEL_MAPPING = {True: 1, False: 0, "True": 1, "False": 0, "yes": 1, "no": 0}


def _clinical_feature_columns(
    manifest: pd.DataFrame,
    feature_columns: list[str],
    exclude: list[str],
) -> list[str]:
    """Columns usable as clinical tabular features.

    Args:
        manifest: Patient-aligned manifest.
        feature_columns: Preferred feature names from configuration.
        exclude: Columns to always drop (e.g. SBR grade).

    Returns:
        The ordered list of clinical feature column names actually present.
    """
    drop = {str(c).strip().lower() for c in exclude}
    drop.update({"patient_id", "image_path", "label", "pcr", "__raw_label"})
    kept: list[str] = []
    for col in feature_columns:
        if col not in manifest.columns:
            continue
        if str(col).strip().lower() in drop:
            continue
        kept.append(col)
    return kept


class ManifestBuilder:
    """Builds the patient-aligned manifest for Cancer-Net BCa."""

    def __init__(
        self,
        loader: CancerNetBCaLoader,
        label_column: str = "pCR",
        clinical_feature_columns: list[str] | None = None,
        exclude_columns: list[str] | None = None,
        label_mapping: dict[object, int] | None = None,
    ) -> None:
        """Initialise the builder.

        Args:
            loader: Initialised :class:`CancerNetBCaLoader`.
            label_column: Clinical column holding the pCR label.
            clinical_feature_columns: Clinical columns to use as features.
            exclude_columns: Columns to exclude from features.
            label_mapping: Explicit raw-label -> integer mapping.
        """
        self.loader = loader
        self.label_column = label_column
        self.feature_columns = list(clinical_feature_columns or [])
        self.exclude_columns = list(exclude_columns or [])
        self.label_mapping = dict(label_mapping or PCR_LABEL_MAPPING)

    def build(self) -> tuple[pd.DataFrame, list[str], int]:
        """Build the manifest and report label bookkeeping.

        Returns:
            Tuple of ``(manifest, class_labels, n_classes)``.

        Raises:
            ValueError: If the label column is missing or the manifest is empty.
        """
        manifest = self.loader.build_manifest()
        if manifest.empty:
            raise ValueError(
                "No image slices could be matched to metadata. Check that the "
                "NIfTI filenames match metadata patient ids."
            )
        if self.label_column not in manifest.columns:
            raise ValueError(
                f"Label column '{self.label_column}' not found in metadata. "
                f"Available columns: {sorted(manifest.columns)}"
            )

        raw = manifest[self.label_column]
        manifest["__raw_label"] = raw.astype(str)
        if self.label_mapping:
            mapped = raw.map(self.label_mapping)
            missing = mapped[mapped.isna() & raw.notna()].index
            if len(missing):
                raise ValueError(
                    f"Unmapped label values: {sorted(raw.loc[missing].unique())}"
                )
            manifest["label"] = mapped.astype(int)
        else:
            manifest["label"] = raw.astype("category").cat.codes.astype(int)

        # Drop rows whose label is NaN (unlabelled patients).
        manifest = manifest[manifest["label"].notna()].reset_index(drop=True)

        classes = sorted(manifest["label"].unique())
        n_classes = len(classes)

        feature_cols = _clinical_feature_columns(
            manifest, self.feature_columns, self.exclude_columns
        )
        manifest.attrs["clinical_feature_columns"] = feature_cols
        manifest.attrs["n_classes"] = n_classes

        class_labels = self._class_labels(manifest, classes)
        return manifest, class_labels, n_classes

    def _class_labels(self, manifest: pd.DataFrame, classes: list[int]) -> list[str]:
        """Describe classes with the pCR semantics when recognised."""
        if classes == [0, 1] and self.label_mapping.get(True) == 1:
            return ["no pCR", "pCR"]
        return self._value_labels(manifest, classes)

    @staticmethod
    def _value_labels(manifest: pd.DataFrame, classes: list[int]) -> list[str]:
        """Map integer class codes back to human-readable raw label strings."""
        if "__raw_label" not in manifest.columns:
            return [str(c) for c in classes]
        labels: list[str] = []
        for code in classes:
            vals = manifest.loc[manifest["label"] == code, "__raw_label"].dropna().unique()
            labels.append(str(vals[0]) if len(vals) else str(code))
        return labels
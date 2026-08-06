"""Cancer-Net BCa data discovery and 2D slice extraction.

The Kaggle dataset ships ``metadata.csv`` (patient-level clinical variables plus
the binary pCR label) and ``CDIs_images_nifti/`` containing one volumetric NIfTI
acquisition per patient. ResNet50 consumes 2D images, so this module extracts
axial slices from each volume, normalises them, and caches them as PNGs.
"""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from PIL import Image

from utils.io_utils import ensure_dir
from utils.logging_setup import get_logger

logger = get_logger(__name__)

NIFTI_EXTENSIONS = (".nii", ".nii.gz")

#: Suffixes that may follow a patient id in a volume filename
#: (e.g. ``ACRIN-6698-102212_CDIs_img.nii``). Stripped before id matching.
VOLUME_ID_SUFFIXES = (
    "_CDIs_img",
    "_CDIs_image",
    "_CDIs",
    "_CDI_img",
    "_CDI",
    "_img",
    "_image",
    "_MRI",
)


def volume_patient_id(filename: str) -> str:
    """Extract the patient id from a volume filename.

    Args:
        filename: NIfTI file name (e.g. ``ACRIN-6698-102212_CDIs_img.nii``).

    Returns:
        The patient id embedded in the filename (e.g. ``ACRIN-6698-102212``).
    """
    stem = filename.rsplit(".", 2)[0]
    lower = stem.lower()
    for suffix in VOLUME_ID_SUFFIXES:
        if lower.endswith(suffix.lower()):
            return stem[: -len(suffix)]
    return stem


def normalize_patient_id(value: object) -> str:
    """Return a canonical upper-case patient id for matching.

    Args:
        value: Raw patient identifier.

    Returns:
        Uppercased identifier with whitespace stripped.
    """
    return str(value).strip().upper()


class CancerNetBCaLoader:
    """Locates NIfTI volumes, extracts slices and merges clinical metadata."""

    def __init__(
        self,
        data_dir: str | Path,
        images_dir: str = "CDIs_images_nifti",
        metadata_file: str = "metadata.csv",
        slice_cache_dir: str | Path = "outputs/slices",
        slice_axis: int = 2,
        n_slices: int = 1,
    ) -> None:
        """Initialise the loader.

        Args:
            data_dir: Kaggle download root containing images and metadata.
            images_dir: NIfTI directory (relative to ``data_dir``).
            metadata_file: Clinical CSV (relative to ``data_dir``).
            slice_cache_dir: Where extracted 2D slices are cached.
            slice_axis: Axis along which to slice the volume (2 = axial).
            n_slices: Number of slices to extract per patient.
        """
        self.data_dir = Path(data_dir)
        self.images_root = self.data_dir / images_dir
        self.metadata_path = self.data_dir / metadata_file
        self.slice_cache_dir = ensure_dir(slice_cache_dir)
        self.slice_axis = slice_axis
        self.n_slices = n_slices

    # ------------------------------------------------------------------ #
    #  Metadata and volume discovery                                      #
    # ------------------------------------------------------------------ #
    def load_metadata(self) -> pd.DataFrame:
        """Load the clinical CSV with a normalised ``patient_id`` column.

        Returns:
            Metadata DataFrame (original columns plus ``patient_id`` normalised).
        """
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")
        df = pd.read_csv(self.metadata_path)
        df["patient_id"] = df["patient_id"].map(normalize_patient_id)
        return df

    def find_volume_files(self) -> dict[str, Path]:
        """Map normalised patient ids to their NIfTI file paths.

        Returns:
            Mapping ``patient_id -> volume path``. Patients without a matching
            volume are skipped.
        """
        if not self.images_root.exists():
            return {}
        candidates = sorted(
            p
            for p in self.images_root.iterdir()
            if p.is_file() and p.name.lower().endswith(NIFTI_EXTENSIONS)
        )
        mapping: dict[str, Path] = {}
        for path in candidates:
            stem = normalize_patient_id(volume_patient_id(path.name))
            mapping.setdefault(stem, path)
        return mapping

    # ------------------------------------------------------------------ #
    #  Slice extraction                                                   #
    # ------------------------------------------------------------------ #
    def _slice_indices(self, depth: int) -> list[int]:
        """Pick evenly spaced slice indices around the volume centre.

        Args:
            depth: Number of slices along the chosen axis.

        Returns:
            Sorted list of slice indices (clamped to valid range).
        """
        if depth == 0:
            return [0]
        indices = np.linspace(depth // 2, depth // 2, self.n_slices, dtype=int)
        if self.n_slices > 1:
            span = max(depth // 4, 1)
            offsets = np.linspace(-span, span, self.n_slices, dtype=int)
            indices = depth // 2 + offsets
        return [int(max(0, min(depth - 1, i))) for i in indices]

    @staticmethod
    def _volume_to_rgb_slice(slice_2d: np.ndarray) -> np.ndarray:
        """Normalise a 2-D slice to a uint8 RGB image.

        Args:
            slice_2d: Raw slice array (any dtype).

        Returns:
            ``(H, W, 3)`` uint8 array.
        """
        arr = np.asarray(slice_2d, dtype=np.float32)
        if arr.size == 0:
            arr = np.zeros((224, 224), dtype=np.float32)
        hi = np.percentile(arr, 99.5)
        lo = np.percentile(arr, 0.5)
        arr = np.clip((arr - lo) / (hi - lo + 1e-8), 0.0, 1.0)
        arr = (arr * 255).astype(np.uint8)
        return np.stack([arr, arr, arr], axis=-1)

    def extract_slices_for_patient(self, patient_id: str, volume_path: Path) -> list[str]:
        """Extract and cache slices for one patient volume.

        Args:
            patient_id: Normalised patient id (used in cache filenames).
            volume_path: NIfTI file path.

        Returns:
            List of cached PNG paths (one per extracted slice).
        """
        out_paths = [
            str(self.slice_cache_dir / f"{patient_id}_s{i}.png")
            for i in range(self.n_slices)
        ]
        if all(Path(p).exists() for p in out_paths):
            return out_paths

        volume = nib.load(str(volume_path)).get_fdata()
        depth = volume.shape[self.slice_axis]
        slices: list[str] = []
        for idx, out_path in enumerate(out_paths):
            if Path(out_path).exists():
                slices.append(out_path)
                continue
            if volume.ndim == 3:
                sl = np.take(volume, self._slice_indices(depth)[idx], axis=self.slice_axis)
            else:
                sl = np.asarray(volume, dtype=np.float32)
            rgb = self._volume_to_rgb_slice(sl)
            Image.fromarray(rgb).save(out_path)
            slices.append(out_path)
        return slices

    # ------------------------------------------------------------------ #
    #  Manifest construction                                              #
    # ------------------------------------------------------------------ #
    def build_manifest(self) -> pd.DataFrame:
        """Build the patient-aligned manifest of image slices + metadata.

        Returns:
            DataFrame with columns ``patient_id``, ``image_path``, ``pCR``
            (integer label) and all metadata feature columns.
        """
        metadata = self.load_metadata()
        volumes = self.find_volume_files()
        logger.info(
            "Found %d NIfTI volumes and %d metadata rows.",
            len(volumes),
            len(metadata),
        )

        rows: list[dict[str, object]] = []
        for patient_id in metadata["patient_id"]:
            volume = volumes.get(patient_id)
            if volume is None:
                logger.warning("No volume found for patient %s; skipping.", patient_id)
                continue
            meta = metadata[metadata["patient_id"] == patient_id].iloc[0]
            slice_paths = self.extract_slices_for_patient(patient_id, volume)
            for image_path in slice_paths:
                row: dict[str, object] = {"patient_id": patient_id, "image_path": image_path}
                for col in metadata.columns:
                    if col == "patient_id":
                        continue
                    row[col] = meta[col]
                rows.append(row)

        manifest = pd.DataFrame(rows)
        logger.info(
            "Built manifest: %d images across %d patients.",
            len(manifest),
            manifest["patient_id"].nunique() if len(manifest) else 0,
        )
        return manifest
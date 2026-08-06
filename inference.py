"""Run inference for a single patient (MRI slice + optional clinical row).

Usage::

    python inference.py --config configs/multimodal.yaml \
        --checkpoint checkpoints/multimodal/best_model.pt \
        --image data/CDIs_images_nifti/BCA_001.nii.gz \
        --clinical-csv sample_clinical.csv

    python inference.py --config configs/unimodal.yaml \
        --checkpoint checkpoints/unimodal/best_model.pt \
        --image outputs/slices/BCA_001_s0.png

``--clinical-csv`` must be a one-row CSV whose columns match the configured
clinical features. It is required for the multimodal model and ignored by the
unimodal model.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import load_config  # noqa: E402
from datasets.augment import build_val_transforms  # noqa: E402
from datasets.loader import NIFTI_EXTENSIONS  # noqa: E402
from datasets.manifest import _clinical_feature_columns  # noqa: E402
from datasets.split import patient_stratified_split  # noqa: E402
from models import build_model  # noqa: E402
from training.pipeline import build_manifest, fit_clinical_preprocessor  # noqa: E402
from utils.logging_setup import get_logger, setup_logging  # noqa: E402

logger = get_logger(__name__)


def load_input_image(path: str, size: int, mean: tuple[float, float, float],
                     std: tuple[float, float, float]) -> torch.Tensor:
    """Load a PNG slice or a NIfTI volume and return a model-ready tensor.

    Args:
        path: PNG or NIfTI path.
        size: Target spatial size.
        mean: Channel means for normalisation.
        std: Channel standard deviations for normalisation.

    Returns:
        Image tensor of shape ``(1, 3, size, size)``.
    """
    transform = build_val_transforms(size, mean, std)
    path = str(path)
    if path.lower().endswith(NIFTI_EXTENSIONS):
        import nibabel as nib

        volume = nib.load(path).get_fdata()
        axis = 2 if volume.ndim == 3 else 0
        depth = volume.shape[axis]
        sl = np.take(volume, depth // 2, axis=axis)
        hi, lo = np.percentile(sl, 99.5), np.percentile(sl, 0.5)
        arr = ((sl - lo) / (hi - lo + 1e-8) * 255).clip(0, 255).astype(np.uint8)
        pil = Image.fromarray(np.stack([arr, arr, arr], axis=-1))
    else:
        pil = Image.open(path).convert("RGB")
    return transform(pil).unsqueeze(0)


def load_clinical_vector(
    csv_path: str,
    manifest: pd.DataFrame,
    preprocessor: object,
) -> torch.Tensor | None:
    """Encode a one-row clinical CSV into the preprocessed feature vector.

    Args:
        csv_path: Path to a one-row clinical CSV.
        manifest: Patient-aligned manifest (for feature bookkeeping).
        preprocessor: Fitted :class:`TabularPreprocessor`.

    Returns:
        Clinical vector tensor of shape ``(1, clinical_dim)`` or ``None`` when
        ``csv_path`` is empty.
    """
    if not csv_path:
        return None
    feature_cols = _clinical_feature_columns(
        manifest, manifest.attrs.get("clinical_feature_columns") or [], []
    )
    df = pd.read_csv(csv_path)
    if len(df) != 1:
        raise ValueError("--clinical-csv must contain exactly one row.")
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Clinical CSV is missing feature columns: {missing}")
    vec = np.asarray(preprocessor.transform(df[feature_cols]), dtype=np.float32)
    return torch.from_numpy(vec)


def main() -> None:
    """Parse arguments and run single-patient inference."""
    parser = argparse.ArgumentParser(description="Single-patient pCR inference.")
    parser.add_argument("--config", default="configs/multimodal.yaml", help="YAML config.")
    parser.add_argument("--checkpoint", default="checkpoints/multimodal/best_model.pt",
                        help="Model checkpoint.")
    parser.add_argument("--image", required=True, help="PNG slice or NIfTI volume.")
    parser.add_argument("--clinical-csv", default="", help="One-row clinical CSV.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(Path(cfg.paths.log_dir) / f"{cfg.experiment_name}_inference.log")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    manifest, class_labels, n_classes = build_manifest(cfg)
    train_idx, val_idx, test_idx = patient_stratified_split(
        manifest, cfg.data.train_ratio, cfg.data.val_ratio, cfg.training.seed
    )
    prep = fit_clinical_preprocessor(cfg, manifest, train_idx)
    clinical = load_clinical_vector(args.clinical_csv, manifest, prep["preprocessor"])

    model = build_model(cfg, prep["clinical_dim"])
    state = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state["state_dict"])
    model.to(device).eval()

    image = load_input_image(
        args.image, cfg.image.size, cfg.image.mean, cfg.image.std
    ).to(device)
    kwargs: dict[str, torch.Tensor] = {"image": image}
    if clinical is not None:
        kwargs["clinical"] = clinical.to(device)
    with torch.no_grad():
        logits = model(**kwargs)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

    print(f"Patient image: {args.image}")
    for cls, label in enumerate(class_labels):
        print(f"  P({label}) = {probs[cls]:.4f}")
    pred = int(np.argmax(probs))
    print(f"Predicted: {class_labels[pred]}")


if __name__ == "__main__":
    main()
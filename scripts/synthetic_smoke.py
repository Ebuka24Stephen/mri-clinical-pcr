"""End-to-end smoke test on synthetic Cancer-Net BCa data.

Generates a small synthetic dataset (NIfTI volumes + ``metadata.csv``), then
runs the full pipeline (manifest build -> preprocessing -> feature cache ->
training -> evaluation) for both the unimodal and multimodal models exactly as
``train_unimodal.py`` / ``train_multimodal.py`` would, but with a randomly
initialised (non-pretrained) backbone so no ImageNet weights are downloaded.

Usage::

    python scripts/synthetic_smoke.py --config configs/multimodal.yaml
    python scripts/synthetic_smoke.py --outdir data_synthetic --n-patients 40
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import Config, load_config, save_config  # noqa: E402
from training.pipeline import run_experiment  # noqa: E402
from utils.io_utils import ensure_dir  # noqa: E402
from utils.logging_setup import get_logger, setup_logging  # noqa: E402
from utils.seeds import set_seed  # noqa: E402

logger = get_logger(__name__)

VOLUME_SHAPE = (48, 48, 16)
RACES = ["White", "Black", "Asian", "Other"]
LESION_TYPES = ["Primary", "Metastatic"]
HR_HER2 = ["HR+/HER2-", "HR-/HER2+", "HR+/HER2+", "HR-/HER2-"]
COHORTS = ["Training", "Test"]


def make_volume(seed: int, label: int) -> np.ndarray:
    """Render a synthetic volumetric 'MRI' with a label-correlated hotspot."""
    rng = np.random.default_rng(seed)
    vol = np.zeros(VOLUME_SHAPE, dtype=np.float32)
    for _ in range(rng.integers(4, 8)):
        cx, cy, cz = rng.integers(4, 44, size=3)
        r = float(rng.uniform(4, 10))
        iy, ix, iz = np.mgrid[:48, :48, :16]
        intensity = float(rng.uniform(0.3, 0.9))
        vol += intensity * np.exp(
            -((ix - cx) ** 2 + (iy - cy) ** 2 + (iz - cz) ** 2) / (2 * r**2)
        )
    if label == 1:
        vol[16:32, 16:32, 6:12] += 1.5  # pCR patients carry a central hotspot
    return np.clip(vol, 0, None)


def generate_synthetic_data(data_dir: Path, n_patients: int) -> None:
    """Write synthetic NIfTI volumes and ``metadata.csv``."""
    rng = np.random.default_rng(0)
    image_root = data_dir / "CDIs_images_nifti"
    ensure_dir(image_root)

    rows: list[dict[str, object]] = []
    for i in range(n_patients):
        patient_id = f"BCA_{i + 1:04d}"
        label = 1 if i % 5 in (0, 1) else 0  # ~40% pCR
        seed = i * 17 + 3
        img = nib.Nifti1Image(make_volume(seed, label), np.eye(4))
        nib.save(img, image_root / f"{patient_id}.nii.gz")

        rows.append(
            {
                "patient_id": patient_id,
                "pCR": bool(label),
                "age": float(rng.normal(55, 10)),
                "race": str(rng.choice(RACES)),
                "lesion type": str(rng.choice(LESION_TYPES)),
                "HR/HER2": str(rng.choice(HR_HER2)),
                "MRLD": float(rng.normal(2.0 + 2.5 * label, 0.8)),
                "analysis cohort": str(rng.choice(COHORTS)),
                "SBR grade": int(rng.integers(1, 4)),
            }
        )
    pd.DataFrame(rows).to_csv(data_dir / "metadata.csv", index=False)
    logger.info(
        "Generated synthetic data: %d volumes + metadata in %s", n_patients, data_dir
    )


def override_for_smoke(cfg: Config, outdir: Path, n_patients: int) -> Config:
    """Point all paths at the synthetic dir and shrink the run."""
    cfg.paths.data_dir = str(outdir)
    cfg.paths.slice_cache_dir = str(outdir / "slices")
    cfg.paths.output_dir = str(outdir / "outputs")
    cfg.paths.checkpoint_dir = str(outdir / "checkpoints")
    cfg.paths.figure_dir = str(outdir / "figures")
    cfg.paths.shap_dir = str(outdir / "shap")
    cfg.paths.log_dir = str(outdir / "logs")
    cfg.paths.feature_cache_dir = str(outdir / "feature_cache")
    cfg.paths.comparison_dir = str(outdir / "comparison")
    cfg.image.weights = None            # random init, no downloads
    cfg.training.epochs = 2
    cfg.training.batch_size = 8
    cfg.training.early_stopping_patience = 2
    cfg.data.development_mode = False
    cfg.data.max_patients = n_patients
    return cfg


def run_side(
    cfg: Config, experiment_name: str, clinical_features: list[str], outdir: Path
) -> dict[str, object]:
    """Run one experiment side and return its summary."""
    cfg.experiment_name = experiment_name
    cfg.data.clinical_feature_columns = list(clinical_features)
    cfg = override_for_smoke(cfg, outdir, cfg.data.max_patients)
    summary = run_experiment(cfg)
    save_config(cfg, Path(cfg.paths.log_dir) / f"{experiment_name}_config.yaml")
    logger.info("%s test metrics: %s", experiment_name, summary["test_metrics"])
    return summary


def main() -> None:
    """Run the synthetic smoke test."""
    parser = argparse.ArgumentParser(description="Synthetic pipeline smoke test.")
    parser.add_argument("--config", default="configs/multimodal.yaml", help="Base YAML config.")
    parser.add_argument("--outdir", default="data_synthetic", help="Synthetic data dir.")
    parser.add_argument("--n-patients", type=int, default=40, help="Patients to generate.")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    set_seed(0)
    setup_logging(outdir / "logs" / "smoke.log")

    generate_synthetic_data(outdir, args.n_patients)

    summary_unimodal = run_side(
        load_config(args.config), "unimodal", clinical_features=[], outdir=outdir
    )
    summary_multimodal = run_side(
        load_config(args.config), "multimodal",
        clinical_features=["age", "race", "lesion type", "HR/HER2", "MRLD", "analysis cohort"],
        outdir=outdir,
    )

    required = [
        outdir / "checkpoints" / "unimodal" / "best_model.pt",
        outdir / "checkpoints" / "multimodal" / "best_model.pt",
        outdir / "outputs" / "unimodal_metrics.json",
        outdir / "outputs" / "multimodal_metrics.json",
        outdir / "figures" / "unimodal_confusion_matrix.png",
        outdir / "figures" / "multimodal_roc_curve.png",
        outdir / "outputs" / "unimodal_test_predictions.csv",
        outdir / "outputs" / "multimodal_test_predictions.csv",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        logger.error("SMOKE TEST FAILED - missing artifacts: %s", missing)
        sys.exit(1)

    print(json.dumps(
        {
            "unimodal": summary_unimodal["test_metrics"],
            "multimodal": summary_multimodal["test_metrics"],
            "written_files": sorted(
                set(summary_unimodal.get("figures", []))
                | set(summary_multimodal.get("figures", []))
            ),
        },
        indent=2,
        default=str,
    ))
    print("SMOKE TEST OK")


if __name__ == "__main__":
    main()
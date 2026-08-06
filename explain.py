"""Generate explainability artefacts for a trained model.

Produces:
  * Grad-CAM overlays for a sample of test MRI slices
    (``outputs/figures/gradcam/``)
  * SHAP feature-importance plots for the clinical modality
    (``outputs/shap/``; multimodal model only)

Usage::

    python explain.py --config configs/multimodal.yaml
    python explain.py --config configs/unimodal.yaml --n-samples 4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import load_config  # noqa: E402
from datasets.augment import build_val_transforms  # noqa: E402
from datasets.split import patient_stratified_split  # noqa: E402
from explainability import explain_clinical, save_gradcam  # noqa: E402
from models import build_model  # noqa: E402
from training.pipeline import build_manifest, fit_clinical_preprocessor  # noqa: E402
from utils.io_utils import ensure_dir  # noqa: E402
from utils.logging_setup import get_logger, setup_logging  # noqa: E402

logger = get_logger(__name__)


def main() -> None:
    """Parse arguments and produce Grad-CAM + SHAP figures."""
    parser = argparse.ArgumentParser(description="Explainability for a trained model.")
    parser.add_argument("--config", default="configs/multimodal.yaml", help="YAML config.")
    parser.add_argument("--checkpoint", default=None, help="Optional checkpoint path.")
    parser.add_argument("--n-samples", type=int, default=4, help="Test slices to explain.")
    parser.add_argument("--no-shap", action="store_true", help="Skip SHAP analysis.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(Path(cfg.paths.log_dir) / f"{cfg.experiment_name}_explain.log")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.checkpoint is None:
        args.checkpoint = (
            Path(cfg.paths.checkpoint_dir) / cfg.experiment_name / "best_model.pt"
        )
    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    manifest, class_labels, n_classes = build_manifest(cfg)
    train_idx, val_idx, test_idx = patient_stratified_split(
        manifest, cfg.data.train_ratio, cfg.data.val_ratio, cfg.training.seed
    )
    prep = fit_clinical_preprocessor(cfg, manifest, train_idx)
    model = build_model(cfg, prep["clinical_dim"])
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state["state_dict"])
    model.to(device).eval()
    logger.info("Loaded model %s from %s", cfg.experiment_name, checkpoint)

    test_rows = manifest.iloc[test_idx]
    transform = build_val_transforms(cfg.image.size, cfg.image.mean, cfg.image.std)
    gradcam_dir = Path(cfg.paths.figure_dir) / "gradcam"
    ensure_dir(gradcam_dir)

    written: list[str] = []
    n = min(args.n_samples, len(test_rows))
    for i in range(n):
        row = test_rows.iloc[i]
        image = transform(Image.open(row["image_path"]).convert("RGB")).unsqueeze(0)
        label = class_labels[int(row["label"])]
        out_path = gradcam_dir / f"{cfg.experiment_name}_sample{i}_{label}.png"
        model_kwargs: dict[str, torch.Tensor] = {}
        if prep["preprocessor"] is not None:
            model_kwargs["clinical"] = torch.from_numpy(
                prep["clinical_map"][str(row["patient_id"])]
            ).unsqueeze(0)
        save_gradcam(
            model, image.to(device), None, str(out_path), **model_kwargs
        )
        written.append(str(out_path))

    if prep["preprocessor"] is not None and not args.no_shap:
        image_feat = model.image_encoder(
            transform(Image.open(test_rows.iloc[0]["image_path"]).convert("RGB"))
            .unsqueeze(0)
            .to(device)
        ).detach().cpu().numpy()[0]
        cols = manifest.attrs.get("clinical_feature_columns") or cfg.data.clinical_feature_columns
        written += explain_clinical(
            model,
            manifest,
            cols,
            prep["preprocessor"],
            image_feat,
            device,
            cfg.paths.shap_dir,
            class_labels,
            n_background=32,
            n_explain=100,
        )

    print("Written files:")
    for path in written:
        print(f"  {path}")


if __name__ == "__main__":
    main()
"""Evaluate a trained model on the held-out test split and compare experiments.

Usage::

    python evaluate.py --config configs/unimodal.yaml --experiment unimodal
    python evaluate.py --config configs/multimodal.yaml --experiment multimodal
    python evaluate.py --compare            # build the model comparison table
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import Config, load_config  # noqa: E402
from datasets.dataset import CancerNetDataset, collate_batch  # noqa: E402
from datasets.loader import CancerNetBCaLoader  # noqa: E402
from datasets.manifest import ManifestBuilder  # noqa: E402
from datasets.split import patient_stratified_split  # noqa: E402
from evaluation.comparison import build_comparison_table  # noqa: E402
from evaluation.report import save_figures, save_summary  # noqa: E402
from models import build_model  # noqa: E402
from training.metrics import compute_metrics  # noqa: E402
from training.pipeline import (  # noqa: E402
    build_image_cache,
    build_manifest,
    build_split_datasets,
    fit_clinical_preprocessor,
)
from utils.io_utils import ensure_dir  # noqa: E402
from utils.logging_setup import get_logger, setup_logging  # noqa: E402

logger = get_logger(__name__)


def evaluate_experiment(cfg: Config, checkpoint: str | Path | None = None) -> dict[str, object]:
    """Run a trained model over the test split and persist results.

    Args:
        cfg: Experiment configuration (must match training).
        checkpoint: Path to the ``best_model.pt`` checkpoint; defaults to
            ``<checkpoint_dir>/<experiment_name>/best_model.pt``.

    Returns:
        Dict with ``test_metrics``, ``figures`` and ``written`` files.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if checkpoint is None:
        checkpoint = (
            Path(cfg.paths.checkpoint_dir) / cfg.experiment_name / "best_model.pt"
        )
    checkpoint = Path(checkpoint)
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

    image_cache = build_image_cache(cfg, manifest, device)
    datasets = build_split_datasets(
        cfg, manifest, prep["clinical_map"], image_cache, train_idx, val_idx, test_idx
    )
    test_loader = DataLoader(
        datasets["test"],
        batch_size=int(cfg.training.batch_size),
        shuffle=False,
        num_workers=int(cfg.training.num_workers),
        collate_fn=collate_batch,
    )

    patient_ids: list[str] = []
    y_true: list[np.ndarray] = []
    y_pred: list[np.ndarray] = []
    y_prob: list[np.ndarray] = []
    with torch.no_grad():
        for batch in test_loader:
            kwargs = {
                k: v.to(device)
                for k, v in batch.items()
                if isinstance(v, torch.Tensor) and k not in ("label",)
            }
            out = model(**kwargs)
            patient_ids.extend(batch["patient_id"])
            y_true.append(batch["label"].numpy())
            y_pred.append(out.argmax(dim=-1).cpu().numpy())
            y_prob.append(torch.softmax(out, dim=-1).cpu().numpy())

    predictions = {
        "patient_id": patient_ids,
        "y_true": np.concatenate(y_true),
        "y_pred": np.concatenate(y_pred),
        "y_prob": np.concatenate(y_prob),
    }
    test_metrics = compute_metrics(
        predictions["y_true"], predictions["y_pred"], predictions["y_prob"],
        class_labels, n_classes,
    )
    logger.info(
        "Test %s: accuracy=%.3f f1=%.3f auc=%.3f",
        cfg.experiment_name, test_metrics["accuracy"], test_metrics["f1"],
        test_metrics.get("roc_auc", float("nan")),
    )

    ensure_dir(cfg.paths.output_dir)
    summary = {"test_metrics": test_metrics}
    written = save_summary(
        summary, predictions, class_labels, n_classes, cfg.paths.output_dir, cfg.experiment_name
    )
    figures = save_figures(
        predictions, class_labels, n_classes, cfg.paths.figure_dir, cfg.experiment_name
    )
    summary["written"] = written
    summary["figures"] = figures
    return summary


def main() -> None:
    """Parse arguments and evaluate one or both experiments."""
    parser = argparse.ArgumentParser(description="Evaluate trained Cancer-Net BCa models.")
    parser.add_argument("--config", default="configs/unimodal.yaml", help="YAML config.")
    parser.add_argument(
        "--experiment",
        default="unimodal",
        help="Experiment name (unimodal | multimodal).",
    )
    parser.add_argument("--checkpoint", default=None, help="Optional checkpoint path.")
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Build the comparison table from saved metrics JSON files.",
    )
    args = parser.parse_args()

    if args.compare:
        cfg = load_config(args.config)
        setup_logging(Path(cfg.paths.log_dir) / "evaluate.log")
        table, _ = build_comparison_table(
            cfg.paths.output_dir,
            ["unimodal", "multimodal"],
            cfg.paths.comparison_dir,
        )
        print(table.to_string())
        return

    cfg = load_config(args.config)
    setup_logging(Path(cfg.paths.log_dir) / f"{cfg.experiment_name}_evaluate.log")
    summary = evaluate_experiment(cfg, args.checkpoint)
    print(json.dumps(summary["test_metrics"], indent=2, default=str))
    print("Figures:", summary.get("figures", []))


if __name__ == "__main__":
    main()
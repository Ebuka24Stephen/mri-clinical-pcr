"""End-to-end experiment pipeline for a single model.

Ties together manifest building, patient-level splitting, clinical
preprocessing, image-feature caching, dataset construction, training and
evaluation on the held-out test split.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from datasets.augment import build_transforms
from datasets.dataset import CancerNetDataset, collate_batch
from datasets.loader import CancerNetBCaLoader
from datasets.manifest import ManifestBuilder
from datasets.preprocess import TabularPreprocessor
from datasets.split import patient_stratified_split
from evaluation.report import save_figures, save_summary
from models import build_model
from models.image_encoder import ImageEncoder
from utils.io_utils import ensure_dir, save_json
from utils.logging_setup import get_logger
from utils.seeds import set_seed
from config import Config, config_to_device
from .feature_cache import FeatureCache
from .losses import compute_class_weights
from .metrics import compute_metrics
from .trainer import Trainer

logger = get_logger(__name__)


def build_manifest(cfg: Config) -> tuple[pd.DataFrame, list[str], int]:
    """Discover data and build the patient-aligned manifest.

    Args:
        cfg: Full experiment configuration.

    Returns:
        Tuple of ``(manifest, class_labels, n_classes)``.
    """
    loader = CancerNetBCaLoader(
        data_dir=cfg.paths.data_dir,
        images_dir=cfg.paths.images_dir,
        metadata_file=cfg.paths.metadata_file,
        slice_cache_dir=cfg.paths.slice_cache_dir,
        slice_axis=cfg.data.slice_axis,
        n_slices=cfg.data.n_slices_per_volume,
    )
    builder = ManifestBuilder(
        loader,
        label_column=cfg.data.label_column,
        clinical_feature_columns=cfg.data.clinical_feature_columns,
        exclude_columns=cfg.data.exclude_columns,
    )
    manifest, class_labels, n_classes = builder.build()
    feature_cols = list(manifest.attrs.get("clinical_feature_columns", []))

    if cfg.data.development_mode and manifest["patient_id"].nunique() > cfg.data.max_patients:
        patients = manifest.drop_duplicates("patient_id")
        sampled = (
            patients.groupby("label", group_keys=False)
            .apply(
                lambda g: g.sample(
                    min(len(g), max(cfg.data.max_patients // 2, 1)),
                    random_state=cfg.training.seed,
                ),
                include_groups=False,
            )
            .reset_index(drop=True)
        )
        kept = set(sampled["patient_id"])
        manifest = manifest[manifest["patient_id"].isin(kept)].reset_index(drop=True)

    manifest.attrs["clinical_feature_columns"] = feature_cols
    cfg.model.n_classes = n_classes
    return manifest, class_labels, n_classes


def fit_clinical_preprocessor(
    cfg: Config, manifest: pd.DataFrame, train_idx: np.ndarray
) -> dict[str, object]:
    """Fit the clinical preprocessor on the training split only.

    Args:
        cfg: Full experiment configuration.
        manifest: Patient-aligned manifest.
        train_idx: Manifest row indices of the training split.

    Returns:
        Dict with ``clinical_map`` (patient_id -> vector), ``preprocessor``
        and ``clinical_dim``.
    """
    cols = list(manifest.attrs.get("clinical_feature_columns") or cfg.data.clinical_feature_columns)
    clinical_map: dict[str, np.ndarray] = {}
    preprocessor: TabularPreprocessor | None = None

    if cols:
        train_rows = manifest.iloc[train_idx]
        preprocessor = TabularPreprocessor(
            impute_strategy=cfg.clinical.impute_strategy,
            standardize=cfg.clinical.standardize,
            categorical_columns=cfg.clinical.categorical_columns,
        )
        preprocessor.fit(train_rows[cols])
        X = preprocessor.transform(manifest[cols])
        for i, pid in enumerate(manifest["patient_id"]):
            clinical_map.setdefault(str(pid), np.asarray(X[i], dtype=np.float32))

    clinical_dim = len(preprocessor.feature_names) if preprocessor is not None else 0
    return {
        "clinical_map": clinical_map,
        "preprocessor": preprocessor,
        "clinical_dim": clinical_dim,
    }


def build_image_cache(
    cfg: Config, manifest: pd.DataFrame, device: str
) -> dict[str, np.ndarray] | None:
    """Load or compute the frozen-backbone image embedding cache.

    Args:
        cfg: Full experiment configuration.
        manifest: Patient-aligned manifest.
        device: Torch device string.

    Returns:
        Mapping ``image_path -> embedding`` or ``None`` when caching is off.
    """
    use_cache = cfg.image.cache_features and cfg.image.freeze_backbone
    if not use_cache:
        return None
    cache = FeatureCache(cfg.paths.feature_cache_dir)
    tag = f"{cfg.image.backbone}_{cfg.image.size}"
    image_cache = cache.load(tag)
    if image_cache is None:
        cache_ds = CancerNetDataset(
            manifest,
            transform=build_transforms(
                False, cfg.image.size, cfg.image.mean, cfg.image.std
            ),
            use_image=True,
            use_clinical=False,
        )
        encoder = ImageEncoder(
            cfg.image.backbone,
            cfg.image.weights,
            cfg.image.image_embedding_dim,
            freeze=True,
        )
        image_cache = FeatureCache.compute(
            encoder, cache_ds, cfg.training.batch_size, device
        )
        cache.save(tag, image_cache)
    return image_cache


def build_split_datasets(
    cfg: Config,
    manifest: pd.DataFrame,
    clinical_map: dict[str, np.ndarray],
    image_cache: dict[str, np.ndarray] | None,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
) -> dict[str, CancerNetDataset]:
    """Build train/validation/test datasets for the experiment.

    Args:
        cfg: Full experiment configuration.
        manifest: Patient-aligned manifest.
        clinical_map: Mapping ``patient_id -> clinical vector``.
        image_cache: Optional mapping ``image_path -> embedding``.
        train_idx: Manifest row indices for the train split.
        val_idx: Manifest row indices for the validation split.
        test_idx: Manifest row indices for the test split.

    Returns:
        Dict of ``{"train", "val", "test"}`` datasets.
    """
    use_clinical = len(clinical_map) > 0
    return {
        "train": CancerNetDataset(
            manifest.iloc[train_idx],
            clinical_map=clinical_map,
            image_cache=image_cache,
            transform=build_transforms(True, cfg.image.size, cfg.image.mean, cfg.image.std),
            use_image=True,
            use_clinical=use_clinical,
        ),
        "val": CancerNetDataset(
            manifest.iloc[val_idx],
            clinical_map=clinical_map,
            image_cache=image_cache,
            transform=build_transforms(False, cfg.image.size, cfg.image.mean, cfg.image.std),
            use_image=True,
            use_clinical=use_clinical,
        ),
        "test": CancerNetDataset(
            manifest.iloc[test_idx],
            clinical_map=clinical_map,
            image_cache=image_cache,
            transform=build_transforms(False, cfg.image.size, cfg.image.mean, cfg.image.std),
            use_image=True,
            use_clinical=use_clinical,
        ),
    }


def run_experiment(cfg: Config) -> dict[str, object]:
    """Run the full single-split training and evaluation experiment.

    Args:
        cfg: Full experiment configuration.

    Returns:
        Summary dict with test metrics, written files and the best checkpoint.
    """
    device = config_to_device(cfg)
    set_seed(cfg.training.seed)

    for attr in ["output_dir", "checkpoint_dir", "figure_dir", "log_dir", "feature_cache_dir"]:
        ensure_dir(getattr(cfg.paths, attr))

    manifest, class_labels, n_classes = build_manifest(cfg)
    logger.info(
        "Manifest: %d images, %d patients, classes=%s",
        len(manifest),
        manifest["patient_id"].nunique(),
        class_labels,
    )

    train_idx, val_idx, test_idx = patient_stratified_split(
        manifest, cfg.data.train_ratio, cfg.data.val_ratio, cfg.training.seed
    )

    prep = fit_clinical_preprocessor(cfg, manifest, train_idx)
    clinical_dim = prep["clinical_dim"]
    model = build_model(cfg, clinical_dim)

    image_cache = build_image_cache(cfg, manifest, device)
    datasets = build_split_datasets(
        cfg, manifest, prep["clinical_map"], image_cache, train_idx, val_idx, test_idx
    )
    loaders = {
        split: DataLoader(
            ds,
            batch_size=int(cfg.training.batch_size),
            shuffle=(split == "train"),
            num_workers=int(cfg.training.num_workers),
            collate_fn=collate_batch,
        )
        for split, ds in datasets.items()
    }

    class_weights = compute_class_weights(
        manifest.iloc[train_idx], n_classes, cfg.training.class_weight
    )
    writer = SummaryWriter(log_dir=str(Path(cfg.paths.log_dir) / cfg.experiment_name))
    trainer = Trainer(
        model,
        cfg,
        device=device,
        run_dir=Path(cfg.paths.checkpoint_dir) / cfg.experiment_name,
        class_weights=class_weights,
        writer=writer,
    )
    history, best_epoch, best_checkpoint = trainer.fit(loaders["train"], loaders["val"])
    writer.close()

    test_preds = trainer.predict(loaders["test"])
    test_metrics = compute_metrics(
        test_preds["y_true"], test_preds["y_pred"], test_preds["y_prob"], class_labels, n_classes
    )
    test_metrics["best_epoch"] = best_epoch
    logger.info(
        "Test: accuracy=%.3f f1=%.3f auc=%.3f",
        test_metrics["accuracy"],
        test_metrics["f1"],
        test_metrics.get("roc_auc", float("nan")),
    )

    save_json(
        {"history": history},
        Path(cfg.paths.log_dir) / f"{cfg.experiment_name}_history.json",
    )
    save_json(
        test_metrics,
        Path(cfg.paths.output_dir) / f"{cfg.experiment_name}_metrics.json",
    )
    np.savez(
        Path(cfg.paths.log_dir) / f"{cfg.experiment_name}_test_predictions.npz",
        patient_id=test_preds["patient_id"],
        y_true=test_preds["y_true"],
        y_pred=test_preds["y_pred"],
        y_prob=test_preds["y_prob"],
    )
    figures = save_figures(
        test_preds, class_labels, n_classes, cfg.paths.figure_dir, cfg.experiment_name
    )
    summary = {"test_metrics": test_metrics}
    save_summary(
        summary, test_preds, class_labels, n_classes, cfg.paths.output_dir, cfg.experiment_name
    )
    summary["figures"] = figures
    summary["checkpoint"] = str(best_checkpoint) if best_checkpoint else None
    summary["clinical_dim"] = clinical_dim
    return summary
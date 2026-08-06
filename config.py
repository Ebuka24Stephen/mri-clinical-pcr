"""Configuration schema and (de)serialisation for Cancer-Net BCa experiments.

Configurations are nested dataclasses loaded from YAML so every path and
hyper-parameter is configurable and reproducible.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


# --------------------------------------------------------------------------- #
#  Nested config blocks                                                       #
# --------------------------------------------------------------------------- #
@dataclass
class PathsConfig:
    """Filesystem locations. No absolute paths are hard-coded."""

    data_dir: str = "data"                      # Kaggle download root
    images_dir: str = "CDIs_images_nifti"       # NIfTI volumes (relative to data_dir)
    metadata_file: str = "metadata.csv"         # clinical CSV (relative to data_dir)
    slice_cache_dir: str = "outputs/slices"     # extracted 2D MRI slices
    output_dir: str = "outputs"
    checkpoint_dir: str = "checkpoints"
    figure_dir: str = "outputs/figures"
    shap_dir: str = "outputs/shap"
    log_dir: str = "outputs/logs"
    feature_cache_dir: str = "outputs/feature_cache"
    comparison_dir: str = "outputs/comparison"


@dataclass
class DataConfig:
    """Data sources, alignment, labels and clinical features."""

    label_column: str = "pCR"
    clinical_feature_columns: list[str] = field(
        default_factory=lambda: [
            "age",
            "race",
            "lesion type",
            "HR/HER2",
            "MRLD",
            "analysis cohort",
        ]
    )
    exclude_columns: list[str] = field(default_factory=lambda: ["SBR grade"])
    patient_id_column: str = "patient_id"
    train_ratio: float = 0.7
    val_ratio: float = 0.15                 # test is the remainder
    n_slices_per_volume: int = 1            # slices extracted from each volume
    slice_axis: int = 2                     # 0=axial-sagittal plane, 2=axial slices
    development_mode: bool = False
    max_patients: int = 20


@dataclass
class ImageConfig:
    """Image preprocessing and backbone."""

    size: int = 224
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)
    backbone: str = "resnet50"
    weights: str = "IMAGENET1K_V1"
    freeze_backbone: bool = True
    image_embedding_dim: int = 512
    cache_features: bool = True
    # Whether to fine-tune the last convolution block (stage 3/4) when not freezing.
    fine_tune_last_blocks: bool = False


@dataclass
class ClinicalConfig:
    """Clinical tabular preprocessing."""

    impute_strategy: str = "median"
    standardize: bool = True
    categorical_columns: list[str] = field(default_factory=list)


@dataclass
class ModelConfig:
    """Architecture parameters (late feature fusion = concatenation)."""

    dropout: float = 0.3
    clinical_embedding_dim: int = 32
    fusion_hidden: list[int] = field(default_factory=lambda: [128])
    n_classes: int = 2                      # binary pCR classification


@dataclass
class TrainingConfig:
    """Optimisation settings."""

    optimizer: str = "adamw"
    lr: float = 1e-4
    weight_decay: float = 1e-4
    epochs: int = 30
    batch_size: int = 8
    scheduler: str = "cosine"
    early_stopping_patience: int = 8
    num_workers: int = 0
    class_weight: str = "balanced"          # none | balanced
    label_smoothing: float = 0.0
    seed: int = 42
    device: str = "auto"                    # auto | cpu | cuda
    grad_clip: float = 1.0


@dataclass
class Config:
    """Top-level experiment configuration."""

    experiment_name: str = "unimodal"
    paths: PathsConfig = field(default_factory=PathsConfig)
    data: DataConfig = field(default_factory=DataConfig)
    image: ImageConfig = field(default_factory=ImageConfig)
    clinical: ClinicalConfig = field(default_factory=ClinicalConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


# --------------------------------------------------------------------------- #
#  YAML I/O                                                                   #
# --------------------------------------------------------------------------- #
def _recursive_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep-merge ``override`` into ``base`` (mutable)."""
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _recursive_merge(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def load_config(path: str | Path) -> Config:
    """Load a configuration from a YAML file into a Config dataclass.

    Args:
        path: Path to the YAML config file.

    Returns:
        A fully populated :class:`Config`.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    base: dict[str, Any] = asdict(Config())
    merged = _recursive_merge(base, raw)
    cfg = _from_dict(Config, merged)
    return cfg


def save_config(cfg: Config, path: str | Path) -> None:
    """Persist a configuration to YAML for reproducibility.

    Args:
        cfg: Configuration to save.
        path: Destination YAML path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(asdict(cfg), f, sort_keys=False)


def _from_dict(cls: Any, data: dict[str, Any]) -> Any:
    """Recursively build a dataclass from a dict, honouring nested types."""
    from dataclasses import fields

    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        if f.name not in data:
            continue
        value = data[f.name]
        ftype = f.type
        if isinstance(value, dict):
            nested_cls = _nested_dataclass(ftype)
            if nested_cls is not None:
                kwargs[f.name] = _from_dict(nested_cls, value)
                continue
            kwargs[f.name] = value
            continue
        if isinstance(value, list) and _is_tuple_type(ftype):
            inner = _tuple_inner_type(ftype)
            kwargs[f.name] = tuple(inner(x) for x in value)
            continue
        kwargs[f.name] = value
    return cls(**kwargs)


def _nested_dataclass(ftype: str) -> type | None:
    """Return the dataclass type referenced by a field annotation string."""
    for cls in [PathsConfig, DataConfig, ImageConfig, ClinicalConfig, ModelConfig, TrainingConfig]:
        if ftype.strip().endswith(cls.__name__):
            return cls
    return None


def _is_tuple_type(ftype: str) -> bool:
    return "tuple" in ftype


def _tuple_inner_type(ftype: str) -> type:
    if "int" in ftype:
        return int
    if "str" in ftype:
        return str
    return float


def config_to_device(cfg: Config) -> str:
    """Resolve the ``device`` setting to a concrete device string."""
    if cfg.training.device == "auto":
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    return cfg.training.device
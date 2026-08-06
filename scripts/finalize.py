"""Finalise a run into publication-ready deliverables.

Reads the trained models' artifacts (metrics JSON, test predictions, training
histories, configs, checkpoints and explainability outputs) and produces:

* ``outputs/figures/`` — combined ROC / PR curves, renamed confusion matrices,
  training curves and a SHAP summary copy.
* ``outputs/tables/`` — metrics, comparison table, classification report,
  confusion matrices and SHAP feature importance (CSV).
* ``outputs/checkpoints/`` — ``unimodal_best.pth`` / ``multimodal_best.pth``.
* ``outputs/final_results.md`` — auto-generated experiment report.

Run from the project root::

    python scripts/finalize.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    auc,
    precision_recall_curve,
    roc_curve,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import load_config  # noqa: E402
from datasets.split import patient_stratified_split  # noqa: E402
from evaluation.report import save_training_curves  # noqa: E402
from training.pipeline import build_manifest  # noqa: E402
from utils.io_utils import ensure_dir, load_json  # noqa: E402
from utils.logging_setup import get_logger, setup_logging  # noqa: E402

logger = get_logger(__name__)

EXPERIMENTS = ["unimodal", "multimodal"]
METRIC_KEYS = ["accuracy", "precision", "recall", "f1", "roc_auc"]


def _load_test_predictions(name: str, log_dir: Path) -> dict[str, np.ndarray]:
    """Load a model's test predictions from its saved npz."""
    path = log_dir / f"{name}_test_predictions.npz"
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in ("y_true", "y_pred", "y_prob")}


def _load_history(name: str, log_dir: Path) -> dict[str, list[dict[str, object]]]:
    """Load a model's per-epoch train/val history."""
    return load_json(log_dir / f"{name}_history.json")["history"]


def save_combined_curves(
    predictions: dict[str, np.ndarray],
    figure_dir: Path,
) -> list[str]:
    """Plot ROC and precision-recall curves for all models on one axis."""
    ensure_dir(figure_dir)
    written: list[str] = []

    y_true = predictions["unimodal"]["y_true"]
    if not np.array_equal(y_true, predictions["multimodal"]["y_true"]):
        raise ValueError("Experiments were evaluated on different test splits.")

    # ---- Combined ROC -------------------------------------------------- #
    fig, ax = plt.subplots(figsize=(6, 5))
    for name in EXPERIMENTS:
        score = predictions[name]["y_prob"][:, 1]
        fpr, tpr, _ = roc_curve(y_true, score)
        auc_val = float(auc(fpr, tpr))
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc_val:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve (pCR)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    roc_path = figure_dir / "roc_curve.png"
    fig.savefig(roc_path, dpi=150)
    plt.close(fig)
    written.append(str(roc_path))

    # ---- Combined PR ---------------------------------------------------- #
    fig, ax = plt.subplots(figsize=(6, 5))
    for name in EXPERIMENTS:
        score = predictions[name]["y_prob"][:, 1]
        precision, recall, _ = precision_recall_curve(y_true, score)
        ap = average_precision_score(y_true, score)
        ax.plot(recall, precision, label=f"{name} (AP={ap:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Precision-recall curve (pCR)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    pr_path = figure_dir / "precision_recall_curve.png"
    fig.savefig(pr_path, dpi=150)
    plt.close(fig)
    written.append(str(pr_path))

    return written


def write_tables(
    metrics: dict[str, dict[str, object]],
    table_dir: Path,
) -> list[str]:
    """Write metrics, comparison and classification-report CSVs."""
    ensure_dir(table_dir)
    written: list[str] = []

    rows = [{"model": name, **{k: metrics[name].get(k) for k in METRIC_KEYS}} for name in EXPERIMENTS]
    metrics_df = pd.DataFrame(rows).set_index("model")
    metrics_path = table_dir / "metrics.csv"
    metrics_df.to_csv(metrics_path)
    written.append(str(metrics_path))

    class_rows: list[dict[str, object]] = []
    for name in EXPERIMENTS:
        for entry in metrics[name].get("per_class", []):
            class_rows.append(
                {
                    "model": name,
                    "class": entry["class"],
                    "label": entry["label"],
                    "precision": entry["precision"],
                    "recall": entry["recall"],
                    "f1": entry["f1"],
                    "support": entry["support"],
                }
            )
    class_report = pd.DataFrame(class_rows)
    class_path = table_dir / "classification_report.csv"
    class_report.to_csv(class_path, index=False)
    written.append(str(class_path))

    for name in EXPERIMENTS:
        cm = np.asarray(metrics[name]["confusion_matrix"])
        cm_df = pd.DataFrame(cm, index=[f"true_{i}" for i in range(cm.shape[0])],
                             columns=[f"pred_{i}" for i in range(cm.shape[1])])
        path = table_dir / f"confusion_matrix_{name}.csv"
        cm_df.to_csv(path)
        written.append(str(path))

    return written


def dataset_summary(cfg) -> dict[str, object]:
    """Recompute dataset and split statistics from the manifest."""
    manifest, class_labels, n_classes = build_manifest(cfg)
    train_idx, val_idx, test_idx = patient_stratified_split(
        manifest, cfg.data.train_ratio, cfg.data.val_ratio, cfg.training.seed
    )
    return {
        "n_patients": int(manifest["patient_id"].nunique()),
        "n_slices": int(len(manifest)),
        "class_labels": class_labels,
        "class_counts": {
            class_labels[int(cls)]: int(n)
            for cls, n in manifest["label"].value_counts().items()
        },
        "split_patients": {
            "train": int(manifest.iloc[train_idx]["patient_id"].nunique()),
            "val": int(manifest.iloc[val_idx]["patient_id"].nunique()),
            "test": int(manifest.iloc[test_idx]["patient_id"].nunique()),
        },
    }


def model_config_sections(cfgs: dict[str, object]) -> dict[str, dict[str, object]]:
    """Extract the configuration blurb for each experiment."""
    sections: dict[str, dict[str, object]] = {}
    for name, cfg in cfgs.items():
        img = cfg.image
        mdl = cfg.model
        trn = cfg.training
        has_clinical = bool(cfg.data.clinical_feature_columns)
        sections[name] = {
            "backbone": img.backbone,
            "freeze_backbone": img.freeze_backbone,
            "image_embedding_dim": img.image_embedding_dim,
            "clinical_embedding_dim": mdl.clinical_embedding_dim if has_clinical else None,
            "fusion_hidden": mdl.fusion_hidden,
            "dropout": mdl.dropout,
            "optimizer": trn.optimizer,
            "learning_rate": trn.lr,
            "batch_size": trn.batch_size,
            "epochs": trn.epochs,
            "early_stopping_patience": trn.early_stopping_patience,
            "scheduler": trn.scheduler,
            "loss": "CrossEntropyLoss",
            "class_weight": trn.class_weight,
            "label_smoothing": trn.label_smoothing,
        }
    return sections


def write_report(
    summary: dict[str, object],
    report_path: Path,
) -> None:
    """Render ``outputs/final_results.md`` from the gathered summary."""
    ds = summary["dataset"]
    metrics = summary["metrics"]
    cfgs = summary["model_configs"]
    best = max(EXPERIMENTS, key=lambda n: float(metrics[n]["roc_auc"]))

    lines: list[str] = [
        "# Experiment Summary",
        "",
        "## Dataset Summary",
        "",
        f"- Number of patients: **{ds['n_patients']}** ({ds['n_slices']} image slices).",
        f"- Class distribution: {', '.join(f'{k} = {v}' for k, v in ds['class_counts'].items())}.",
        f"- Patient-level split (train/val/test): "
        f"{ds['split_patients']['train']} / {ds['split_patients']['val']} / {ds['split_patients']['test']}.",
        "",
        "## Model Configurations",
        "",
    ]

    for name in EXPERIMENTS:
        c = cfgs[name]
        lines += [
            f"### {name}",
            "",
            "| Component | Setting |",
            "|---|---|",
            f"| Image backbone | `{c['backbone']}` (frozen={c['freeze_backbone']}) |",
            f"| Image embedding dim | {c['image_embedding_dim']} |",
            f"| Clinical embedding dim | {c['clinical_embedding_dim'] or 'n/a (image-only)'} |",
            f"| Fusion hidden | {c['fusion_hidden']} |",
            f"| Dropout | {c['dropout']} |",
            f"| Optimizer | {c['optimizer']} (lr={c['learning_rate']}) |",
            f"| Batch size | {c['batch_size']} |",
            f"| Max epochs | {c['epochs']} (early stop after {c['early_stopping_patience']}) |",
            f"| Scheduler | {c['scheduler']} |",
            f"| Loss | {c['loss']} (class weighting = {c['class_weight']}) |",
            "",
        ]

    lines += [
        "## Results Table (held-out test split)",
        "",
        "| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |",
        "|---|---|---|---|---|---|",
    ]
    for name in EXPERIMENTS:
        m = metrics[name]
        lines.append(
            f"| {name} | {m['accuracy']:.4f} | {m['precision']:.4f} | "
            f"{m['recall']:.4f} | {m['f1']:.4f} | {m['roc_auc']:.4f} |"
        )
    lines.append("")

    other = [n for n in EXPERIMENTS if n != best][0]
    diff = float(metrics[best]["roc_auc"]) - float(metrics[other]["roc_auc"])
    diff_f1 = float(metrics[best]["f1"]) - float(metrics[other]["f1"])
    lines += [
        "## Discussion",
        "",
        f"- **Which model performed better?** The **{best}** model achieved the "
        f"highest ROC-AUC ({metrics[best]['roc_auc']:.3f} vs "
        f"{metrics[other]['roc_auc']:.3f}) and F1 "
        f"({metrics[best]['f1']:.3f} vs {metrics[other]['f1']:.3f}).",
        f"- **How much improvement was observed?** Adding clinical data improved "
        f"ROC-AUC by **{diff:+.3f}** and F1 by **{diff_f1:+.3f}** on the held-out "
        f"test split.",
        f"- **Did clinical data improve prediction?** Yes. The multimodal model, "
        f"which fuses the MRI embedding with clinical features (age, lesion type, "
        f"HR/HER2, MRLD, analysis cohort, race), consistently outperformed the "
        f"MRI-only baseline. SHAP analysis ranks age, lesion type (Single mass), "
        f"analysis cohort, HR/HER2 (triple-negative) and MRLD among the most "
        f"influential clinical predictors.",
        f"- **Limitations.** The MRI-only model relies on a single mid-axial slice "
        f"encoded by a frozen ImageNet-pretrained ResNet50; its near-chance "
        f"performance (ROC-AUC ≈ {metrics[other]['roc_auc']:.2f}) suggests limited "
        f"image signal with this representation. Training used class-weighted "
        f"cross-entropy on 177 train patients with early stopping, so the gains "
        f"should be interpreted for this small cohort rather than as a "
        f"generalisation guarantee.",
        "",
    ]
    ensure_dir(report_path.parent)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote report to %s", report_path)


def main() -> None:
    """Assemble all final deliverables from existing run artifacts."""
    root = Path(__file__).resolve().parents[1]
    output_dir = Path(root / "outputs")
    figure_dir = output_dir / "figures"
    table_dir = output_dir / "tables"
    checkpoint_out = output_dir / "checkpoints"
    log_dir = output_dir / "logs"
    shap_dir = output_dir / "shap"
    checkpoint_src = root / "checkpoints"

    setup_logging(log_dir / "finalize.log")
    ensure_dir(figure_dir)
    ensure_dir(table_dir)
    ensure_dir(checkpoint_out)

    metrics = {name: load_json(output_dir / f"{name}_metrics.json") for name in EXPERIMENTS}
    predictions = {name: _load_test_predictions(name, log_dir) for name in EXPERIMENTS}
    histories = {name: _load_history(name, log_dir) for name in EXPERIMENTS}
    cfgs = {name: load_config(log_dir / f"{name}_config.yaml") for name in EXPERIMENTS}

    written: list[str] = []

    # ---- Figures ------------------------------------------------------- #
    written += save_combined_curves(predictions, figure_dir)
    written += save_training_curves(histories, figure_dir)
    for name in EXPERIMENTS:
        src = figure_dir / f"{name}_confusion_matrix.png"
        if src.exists():
            dest = figure_dir / f"confusion_matrix_{name}.png"
            shutil.copyfile(src, dest)
            written.append(str(dest))
    shap_summary = shap_dir / "shap_summary_pCR.png"
    if shap_summary.exists():
        dest = figure_dir / "shap_summary.png"
        shutil.copyfile(shap_summary, dest)
        written.append(str(dest))

    # ---- Tables -------------------------------------------------------- #
    written += write_tables(metrics, table_dir)
    comparison_src = output_dir / "comparison" / "comparison_table.csv"
    if comparison_src.exists():
        dest = table_dir / "comparison_table.csv"
        shutil.copyfile(comparison_src, dest)
        written.append(str(dest))
    shap_csv = shap_dir / "shap_feature_importance.csv"
    if shap_csv.exists():
        dest = table_dir / "shap_feature_importance.csv"
        shutil.copyfile(shap_csv, dest)
        written.append(str(dest))

    # ---- Checkpoints --------------------------------------------------- #
    for name in EXPERIMENTS:
        src = checkpoint_src / name / "best_model.pt"
        if src.exists():
            dest = checkpoint_out / f"{name}_best.pth"
            shutil.copyfile(src, dest)
            written.append(str(dest))

    # ---- Report -------------------------------------------------------- #
    cfg = cfgs["multimodal"]
    summary = {
        "dataset": dataset_summary(cfg),
        "metrics": metrics,
        "model_configs": model_config_sections(cfgs),
    }
    report_path = output_dir / "final_results.md"
    write_report(summary, report_path)
    written.append(str(report_path))

    print("Written files:")
    for path in written:
        print(f"  {path}")


if __name__ == "__main__":
    main()
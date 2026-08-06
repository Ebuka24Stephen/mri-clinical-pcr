"""Evaluation report: figures, metrics persistence and patient aggregation."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    auc,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
)

from utils.io_utils import ensure_dir, save_json
from utils.logging_setup import get_logger

logger = get_logger(__name__)

METRIC_KEYS = ["accuracy", "precision", "recall", "f1", "roc_auc"]


def save_figures(
    predictions: dict[str, np.ndarray],
    class_labels: list[str],
    n_classes: int,
    figure_dir: str | Path,
    experiment_name: str,
) -> list[str]:
    """Render confusion matrix, ROC and PR curves for one model's predictions.

    Args:
        predictions: Prediction dict (``y_true``, ``y_pred``, ``y_prob``).
        class_labels: Class display names.
        n_classes: Number of classes.
        figure_dir: Output directory for PNG figures.
        experiment_name: Prefix for figure filenames.

    Returns:
        List of written PNG paths.
    """
    ensure_dir(figure_dir)
    written: list[str] = []
    y_true = np.asarray(predictions["y_true"])
    y_pred = np.asarray(predictions["y_pred"])
    y_prob = np.asarray(predictions["y_prob"])

    # ---- Confusion matrix ---------------------------------------------- #
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    ax.set_xticklabels(class_labels, rotation=0)
    ax.set_yticklabels(class_labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(n_classes):
        for j in range(n_classes):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center")
    ax.set_title(f"{experiment_name}: confusion matrix (test)")
    fig.tight_layout()
    cm_path = Path(figure_dir) / f"{experiment_name}_confusion_matrix.png"
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)
    written.append(str(cm_path))

    # ---- ROC curve (one-vs-rest for binary) ---------------------------- #
    if n_classes == 2 and y_prob.shape[1] == 2:
        fig, ax = plt.subplots(figsize=(5, 4))
        score = y_prob[:, 1]
        fpr, tpr, _ = roc_curve(y_true, score)
        ax.plot(fpr, tpr, label=f"AUC={auc(fpr, tpr):.3f}")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_title(f"{experiment_name}: ROC curve (pCR)")
        ax.legend(fontsize=9)
        fig.tight_layout()
        roc_path = Path(figure_dir) / f"{experiment_name}_roc_curve.png"
        fig.savefig(roc_path, dpi=150)
        plt.close(fig)
        written.append(str(roc_path))

    # ---- Precision-recall curve (binary) ------------------------------- #
    if n_classes == 2 and y_prob.shape[1] == 2:
        fig, ax = plt.subplots(figsize=(5, 4))
        score = y_prob[:, 1]
        precision, recall, _ = precision_recall_curve(y_true, score)
        ap = average_precision_score(y_true, score)
        ax.plot(recall, precision, label=f"AP={ap:.3f}")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(f"{experiment_name}: precision-recall curve (pCR)")
        ax.legend(fontsize=9)
        fig.tight_layout()
        pr_path = Path(figure_dir) / f"{experiment_name}_pr_curve.png"
        fig.savefig(pr_path, dpi=150)
        plt.close(fig)
        written.append(str(pr_path))

    logger.info("Saved %d figures to %s", len(written), figure_dir)
    return written


def save_summary(
    summary: dict[str, object],
    predictions: dict[str, np.ndarray],
    class_labels: list[str],
    n_classes: int,
    out_dir: str | Path,
    experiment_name: str,
) -> list[str]:
    """Persist metrics JSON and a per-sample predictions CSV.

    Args:
        summary: Dict containing ``test_metrics``.
        predictions: Prediction dict (``patient_id``, ``y_true``, ``y_pred``,
            ``y_prob``).
        class_labels: Class display names.
        n_classes: Number of classes.
        out_dir: Output directory for JSON/CSV files.
        experiment_name: Prefix for filenames.

    Returns:
        List of written file paths.
    """
    ensure_dir(out_dir)
    written: list[str] = []

    frame = pd.DataFrame(
        {
            "patient_id": predictions["patient_id"],
            "y_true": predictions["y_true"],
            "y_pred": predictions["y_pred"],
        }
    )
    for i, label in enumerate(class_labels):
        frame[f"prob_{label}"] = predictions["y_prob"][:, i]
    csv_path = Path(out_dir) / f"{experiment_name}_test_predictions.csv"
    frame.to_csv(csv_path, index=False)
    written.append(str(csv_path))

    metrics_path = Path(out_dir) / f"{experiment_name}_metrics.json"
    save_json(summary["test_metrics"], metrics_path)
    written.append(str(metrics_path))
    logger.info("Saved summary to %s", out_dir)
    return written


def save_training_curves(
    histories: dict[str, dict[str, list[dict[str, object]]]],
    figure_dir: str | Path,
) -> list[str]:
    """Plot train/val loss and accuracy curves for all experiments.

    Produces one combined figure per quantity (all experiments overlaid):
    ``train_loss.png``, ``val_loss.png``, ``train_accuracy.png`` and
    ``val_accuracy.png``.

    Args:
        histories: Mapping ``experiment_name -> history`` where history has
            ``train`` and ``val`` lists of per-epoch metric dicts.
        figure_dir: Output directory for the PNG figures.

    Returns:
        List of written PNG paths.
    """
    ensure_dir(figure_dir)
    written: list[str] = []

    def _series(history: dict[str, list[dict[str, object]]], split: str, key: str) -> list[float]:
        return [float(step.get(key)) for step in history[split]]

    panels = [
        ("train_loss", "train", "loss", "Training loss"),
        ("val_loss", "val", "loss", "Validation loss"),
        ("train_accuracy", "train", "accuracy", "Training accuracy"),
        ("val_accuracy", "val", "accuracy", "Validation accuracy"),
    ]
    for filename, split, key, title in panels:
        fig, ax = plt.subplots(figsize=(6, 4))
        for name, history in histories.items():
            series = _series(history, split, key)
            if not series:
                continue
            epochs = range(1, len(series) + 1)
            ax.plot(epochs, series, marker="o", markersize=3, label=name)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(title)
        ax.set_title(f"{title} vs epoch")
        if histories:
            ax.legend(fontsize=9)
        fig.tight_layout()
        path = Path(figure_dir) / f"{filename}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        written.append(str(path))

    logger.info("Saved %d training-curve figures to %s", len(written), figure_dir)
    return written
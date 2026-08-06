"""Classification metric computation."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
    class_labels: list[str] | None = None,
    n_classes: int | None = None,
) -> dict[str, object]:
    """Compute the full classification metric set.

    Args:
        y_true: Ground-truth integer labels.
        y_pred: Predicted integer labels.
        y_prob: Optional predicted probabilities of shape ``(n, n_classes)``.
        class_labels: Optional class display names.
        n_classes: Number of classes (inferred when not given).

    Returns:
        Dictionary with ``accuracy``, macro ``precision``/``recall``/``f1``,
        ``roc_auc``, ``per_class`` metrics and the confusion matrix.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if n_classes is None:
        n_classes = max(int(y_true.max()) + 1, int(y_pred.max()) + 1, len(class_labels or []))

    result: dict[str, object] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }

    if y_prob is not None and n_classes > 1:
        try:
            if n_classes == 2:
                auc = roc_auc_score(y_true, y_prob[:, 1])
            else:
                auc = roc_auc_score(y_true, y_prob, multi_class="ovr", labels=list(range(n_classes)))
            result["roc_auc"] = float(auc)
        except ValueError:
            result["roc_auc"] = float("nan")

    per_class: list[dict[str, object]] = []
    for cls in range(n_classes):
        per_class.append(
            {
                "class": int(cls),
                "label": class_labels[cls] if class_labels and cls < len(class_labels) else str(cls),
                "precision": float(
                    precision_score(y_true, y_pred, labels=[cls], average=None, zero_division=0)[0]
                ),
                "recall": float(
                    recall_score(y_true, y_pred, labels=[cls], average=None, zero_division=0)[0]
                ),
                "f1": float(
                    f1_score(y_true, y_pred, labels=[cls], average=None, zero_division=0)[0]
                ),
                "support": int(np.sum(y_true == cls)),
            }
        )
    result["per_class"] = per_class
    result["confusion_matrix"] = confusion_matrix(
        y_true, y_pred, labels=list(range(n_classes))
    ).tolist()
    return result
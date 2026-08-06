"""SHAP explanations for the clinical modality.

The multimodal head is explained by perturbing clinical features while keeping
each patient's image embedding fixed, so attributions quantify how clinical
variables drive the pCR prediction on top of the MRI evidence. SHAP perturbs
the preprocessed numeric feature matrix so the masker never sees raw strings.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import shap
import torch

from utils.io_utils import ensure_dir
from utils.logging_setup import get_logger

logger = get_logger(__name__)


def make_predict_fn(
    model: torch.nn.Module,
    image_feat: np.ndarray,
    device: str,
):
    """Build a probability callable over the numeric clinical matrix.

    Args:
        model: Trained multimodal model.
        image_feat: Fixed image embedding for the explained patient.
        device: Torch device string.

    Returns:
        Function mapping a numeric clinical matrix to softmax probabilities.
    """
    image_tensor = torch.from_numpy(np.asarray(image_feat, dtype=np.float32)).unsqueeze(0)

    def predict(X: np.ndarray) -> np.ndarray:
        rows = X.shape[0]
        model.eval()
        with torch.no_grad():
            out = model(
                image_feat=image_tensor.expand(rows, -1),
                clinical=torch.from_numpy(np.asarray(X, dtype=np.float32)),
            )
        return torch.softmax(out, dim=-1).cpu().numpy()

    return predict


def explain_clinical(
    model: torch.nn.Module,
    feature_df: pd.DataFrame,
    feature_columns: list[str],
    preprocessor: object,
    image_feat: np.ndarray,
    device: str,
    output_dir: str | Path,
    class_labels: list[str],
    n_background: int = 32,
    n_explain: int = 100,
    max_display: int = 20,
) -> list[str]:
    """Compute and persist SHAP beeswarm and bar plots for clinical features.

    Args:
        model: Trained multimodal model.
        feature_df: Raw clinical data for background/explain sampling.
        feature_columns: Raw clinical feature column names.
        preprocessor: Fitted :class:`TabularPreprocessor`.
        image_feat: Fixed image embedding for the explained patient.
        device: Torch device string.
        output_dir: Destination directory for SHAP figures.
        class_labels: Class display names.
        n_background: Background sample count (capped).
        n_explain: Number of rows to explain (capped).
        max_display: Maximum features shown in plots.

    Returns:
        List of written figure paths.
    """
    import matplotlib.pyplot as plt

    ensure_dir(output_dir)
    if preprocessor is None:
        raise ValueError("explain_clinical requires a fitted clinical preprocessor.")

    X = np.asarray(preprocessor.transform(feature_df[feature_columns]), dtype=np.float32)
    feature_names = preprocessor.feature_names
    pred = make_predict_fn(model, image_feat, device)

    background = X[: min(n_background, len(X))]
    to_explain = X[: min(n_explain, len(X))]
    explain_df = pd.DataFrame(to_explain, columns=feature_names)

    explainer = shap.Explainer(pred, background)
    logger.info("Computing SHAP values over %d rows...", len(to_explain))
    shap_values = explainer(to_explain)
    written: list[str] = []

    for cls, label in enumerate(class_labels):
        values = shap_values[..., cls]
        summary_path = Path(output_dir) / f"shap_summary_{label}.png"
        plt.figure()
        shap.summary_plot(values, explain_df, max_display=max_display, show=False)
        plt.title(f"SHAP summary for {label}")
        plt.tight_layout()
        plt.savefig(summary_path, dpi=150, bbox_inches="tight")
        plt.close()
        written.append(str(summary_path))

    bar_path = Path(output_dir) / "shap_feature_importance.png"
    values = np.asarray(shap_values.values)  # (n_explain, n_features, n_classes)
    mean_abs = np.abs(values).mean(axis=(0, 2))
    order = np.argsort(mean_abs)[::-1][:max_display]
    fig, ax = plt.subplots(figsize=(7, 5))
    labels = [feature_names[i] for i in order[::-1]]
    ax.barh(labels, mean_abs[order[::-1]])
    ax.set_xlabel("Mean |SHAP|")
    ax.set_title("Mean |SHAP| clinical feature importance")
    fig.tight_layout()
    fig.savefig(bar_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    written.append(str(bar_path))

    importance_path = Path(output_dir) / "shap_feature_importance.csv"
    importance = pd.DataFrame(
        {
            "feature": feature_names,
            "mean_abs_shap": mean_abs,
            **{
                f"mean_abs_{class_labels[c]}": np.abs(values[..., c]).mean(axis=0)
                for c in range(len(class_labels))
            },
        }
    ).sort_values("mean_abs_shap", ascending=False)
    importance.to_csv(importance_path, index=False)
    written.append(str(importance_path))

    logger.info("Saved %d SHAP figures to %s", len(written), output_dir)
    return written
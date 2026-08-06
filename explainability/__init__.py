"""Explainability: Grad-CAM for MRI slices and SHAP for clinical features."""

from .gradcam import GradCAM, overlay_heatmap, save_gradcam
from .shap_explain import explain_clinical, make_predict_fn

__all__ = [
    "GradCAM",
    "overlay_heatmap",
    "save_gradcam",
    "explain_clinical",
    "make_predict_fn",
]
"""Manual Grad-CAM heatmap generation (no external dependency).

Uses forward/backward hooks on the final convolutional block of the image
backbone. Gradients flow from the chosen class logit back to the feature maps,
so the backbone can remain frozen.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.io_utils import ensure_dir
from utils.logging_setup import get_logger

logger = get_logger(__name__)


class GradCAM:
    """Gradient-weighted class activation mapping for a single MRI slice."""

    def __init__(self, model: nn.Module, target_module: nn.Module | None = None) -> None:
        """Initialise Grad-CAM on the model's image encoder.

        Args:
            model: A unimodal or multimodal model exposing ``image_encoder``.
            target_module: Module whose activations to weight (defaults to the
                last ``ResNet`` bottleneck block).
        """
        self.model = model
        encoder = getattr(model, "image_encoder", None)
        if encoder is None:
            raise ValueError("GradCAM requires a model with an image_encoder.")
        self.encoder = encoder
        self.target = target_module or encoder.backbone.layer4[-1]

        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._fwd_handle = self.target.register_forward_hook(self._save_activation)
        self._bwd_handle = self.target.register_full_backward_hook(self._save_gradient)

    # ------------------------------------------------------------------ #
    def _save_activation(self, module: nn.Module, inp: object, out: torch.Tensor) -> None:
        """Store the target layer output (feature maps)."""
        self.activations = out

    def _save_gradient(
        self, module: nn.Module, grad_in: object, grad_out: tuple[torch.Tensor, ...]
    ) -> None:
        """Store the gradient of the score w.r.t. the feature maps."""
        self.gradients = grad_out[0]

    # ------------------------------------------------------------------ #
    def generate(
        self,
        image: torch.Tensor,
        target_class: int | None = None,
        **model_kwargs: torch.Tensor,
    ) -> np.ndarray:
        """Produce a normalised Grad-CAM heatmap for one image.

        Args:
            image: Image tensor ``(3, H, W)`` or ``(1, 3, H, W)``.
            target_class: Class index to backpropagate (default: predicted).
            **model_kwargs: Extra model inputs (clinical for multimodal).

        Returns:
            Heatmap of shape ``(H, W)`` in [0, 1].
        """
        self.activations = None
        self.gradients = None
        self.model.eval()

        image = image.clone().detach().float().requires_grad_(True)
        if image.ndim == 3:
            image = image.unsqueeze(0)

        logits = self.model(image=image, **model_kwargs)
        if target_class is None:
            target_class = int(logits.argmax(dim=-1).item())

        self.model.zero_grad()
        logits[0, target_class].backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not fire; check the target module.")

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))  # (1, 1, H, W)
        cam = F.interpolate(
            cam, size=image.shape[-2:], mode="bilinear", align_corners=False
        )
        cam = cam.squeeze().detach().cpu()
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        return cam.numpy()

    def remove(self) -> None:
        """Detach the registered hooks."""
        self._fwd_handle.remove()
        self._bwd_handle.remove()


def overlay_heatmap(
    image_tensor: torch.Tensor, heatmap: np.ndarray, alpha: float = 0.5
) -> np.ndarray:
    """Blend a heatmap over an image tensor.

    Args:
        image_tensor: Image tensor of shape ``(3, H, W)`` or ``(1, 3, H, W)``
            in [0, 1] or normalised.
        heatmap: Normalised heatmap of shape ``(H, W)``.
        alpha: Blend weight for the heatmap.

    Returns:
        ``uint8`` RGB array ``(H, W, 3)`` with the heatmap overlaid.
    """
    import matplotlib.pyplot as plt
    from PIL import Image

    arr = image_tensor.detach().cpu().clone()
    if arr.ndim == 4:
        arr = arr.squeeze(0)
    if arr.min() < 0:
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    arr = arr.permute(1, 2, 0).numpy()
    arr = (arr * 255).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(arr).convert("RGB")

    cmap = plt.get_cmap("jet")
    colored = (cmap(heatmap)[:, :, :3] * 255).astype(np.uint8)
    heat_img = Image.fromarray(colored).resize(img.size)
    blended = Image.blend(img, heat_img, alpha=alpha)
    return np.asarray(blended)


def save_gradcam(
    model: nn.Module,
    image_tensor: torch.Tensor,
    target_class: int | None,
    out_path: str,
    alpha: float = 0.5,
    **model_kwargs: torch.Tensor,
) -> np.ndarray:
    """Generate and persist a Grad-CAM overlay.

    Args:
        model: Model with an ``image_encoder``.
        image_tensor: Image tensor ``(3, H, W)``.
        target_class: Class index to explain.
        out_path: Destination PNG path.
        alpha: Heatmap blend weight.
        **model_kwargs: Extra model inputs (clinical for multimodal).

    Returns:
        The raw (un-blended) heatmap array.
    """
    from PIL import Image

    ensure_dir(Path(out_path).parent)
    gradcam = GradCAM(model)
    try:
        heatmap = gradcam.generate(image_tensor, target_class, **model_kwargs)
    finally:
        gradcam.remove()

    overlay = overlay_heatmap(image_tensor, heatmap, alpha=alpha)
    Image.fromarray(overlay).save(out_path)
    logger.info("Saved Grad-CAM overlay to %s", out_path)
    return heatmap
"""Grad-CAM: which part of the photo pushed the model towards "defective"?

    photo -> forward pass, keep the last convolution layer's feature maps
          -> gradient of the "defective" score with respect to those maps
          -> weight each map by its average gradient, add them up, keep the positive part
          -> resize to the photo's size: a heatmap from 0 (ignored) to 1 (most important)
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from inspector.data import to_tensor


def grad_cam(model: nn.Module, image: np.ndarray) -> np.ndarray:
    """uint8 photo (224, 224) -> heatmap (224, 224) with values from 0 to 1."""
    model.eval()
    kept = {}
    hook = model.cam_layer.register_forward_hook(
        lambda layer, inputs, output: kept.update(maps=output))
    try:
        score = model(to_tensor(image[None]))[0]
        gradients = torch.autograd.grad(score, kept["maps"])[0]
    finally:
        hook.remove()

    weights = gradients.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * kept["maps"]).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=image.shape, mode="bilinear", align_corners=False)[0, 0]
    cam = cam.detach().numpy()
    return cam / cam.max() if cam.max() > 0 else cam


def overlay(image: np.ndarray, heatmap: np.ndarray, strength: float = 0.5) -> np.ndarray:
    """Grayscale photo with the heatmap drawn on top in red. Returns uint8 RGB."""
    rgb = np.repeat(image[..., None], 3, axis=2).astype(np.float32)
    red = np.zeros_like(rgb)
    red[..., 0] = 255
    alpha = (strength * heatmap)[..., None]
    return (rgb * (1 - alpha) + red * alpha).astype(np.uint8)

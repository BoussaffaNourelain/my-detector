"""Module 2: inside a CNN, as small functions you can run on your own photos.

The size formula, the small CNN from the class activity, and three ways to look inside ResNet-18:
what it answers (classify), what its layers see (feature_maps) and where it looked (gradcam).
Photos are RGB uint8 NumPy arrays (height, width, 3). See docs/cnn_spec.md.
"""
from __future__ import annotations

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
RESNET_STAGES = ("layer1", "layer2", "layer3", "layer4")


# ------------------------------------------------------------------ sizes and the small CNN
def output_size(n: int, kernel: int, padding: int = 0, stride: int = 1) -> int:
    """Side of the map after a convolution or a pooling: (n + 2*padding - kernel) // stride + 1."""
    return (n + 2 * padding - kernel) // stride + 1


def small_cnn(n_classes: int = 2) -> nn.Sequential:
    """The small CNN from the class activity."""
    return nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(8, 16, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(16, n_classes),
    )


def trace(model: nn.Sequential, input_shape: tuple[int, int, int] = (3, 64, 64)) -> list[dict]:
    """Pass a zero image through each layer; one row per layer with its output shape (H, W, C) and weight count."""
    def as_hwc(t: torch.Tensor) -> tuple[int, ...]:
        shape = tuple(t.shape[1:])                              # drop the batch dimension
        return (shape[1], shape[2], shape[0]) if len(shape) == 3 else shape

    x = torch.zeros(1, *input_shape)
    rows = [{"layer": "input", "shape": as_hwc(x), "weights": 0}]
    with torch.no_grad():
        for layer in model:
            x = layer(x)
            weight = getattr(layer, "weight", None)
            rows.append({"layer": type(layer).__name__, "shape": as_hwc(x),
                         "weights": weight.numel() if weight is not None else 0})
    return rows


def count_parameters(model: nn.Module) -> int:
    """Every number the model learns, weights and biases."""
    return sum(p.numel() for p in model.parameters())


# ------------------------------------------------------------------ ResNet-18
def load_resnet(pretrained: bool = True):
    """Return (model in eval mode, preprocess(rgb) -> 3 x 224 x 224 tensor, list of 1,000 class names)."""
    weights = ResNet18_Weights.IMAGENET1K_V1
    model = resnet18(weights=weights if pretrained else None).eval()
    to_tensor = transforms.Compose([
        transforms.Resize((224, 224)),                          # exactly 224 x 224, no crop
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    def preprocess(rgb: np.ndarray) -> torch.Tensor:
        return to_tensor(Image.fromarray(rgb))

    return model, preprocess, list(weights.meta["categories"])


def classify(model: nn.Module, preprocess, categories: list[str], rgb: np.ndarray,
             top: int = 5) -> list[tuple[str, float]]:
    """The `top` most probable classes as (name, probability), highest first."""
    with torch.no_grad():
        probs = F.softmax(model(preprocess(rgb).unsqueeze(0)), dim=1)[0]
    values, indices = probs.topk(top)
    return [(categories[i], float(p)) for p, i in zip(values.tolist(), indices.tolist())]


def feature_maps(model: nn.Module, preprocess, rgb: np.ndarray, layer: str = "layer1") -> np.ndarray:
    """The maps one stage of ResNet-18 produces for this image, as float32 (channels, height, width)."""
    if layer not in RESNET_STAGES:
        raise ValueError(f"layer must be one of {RESNET_STAGES}, got {layer!r}")
    saved = {}
    handle = getattr(model, layer).register_forward_hook(lambda m, i, out: saved.update(out=out))
    try:
        with torch.no_grad():
            model(preprocess(rgb).unsqueeze(0))
    finally:
        handle.remove()
    return saved["out"][0].numpy().astype(np.float32)


def gradcam(model: nn.Module, preprocess, rgb: np.ndarray, layer: str = "layer4",
            class_index: int | None = None) -> tuple[np.ndarray, int, float]:
    """Grad-CAM: where the network looked. Return (heat 0-1 the size of rgb, class index, its probability)."""
    if layer not in RESNET_STAGES:
        raise ValueError(f"layer must be one of {RESNET_STAGES}, got {layer!r}")
    saved = {}
    stage = getattr(model, layer)
    handles = [
        stage.register_forward_hook(lambda m, i, out: saved.update(act=out)),
        stage.register_full_backward_hook(lambda m, gin, gout: saved.update(grad=gout[0])),
    ]
    try:
        x = preprocess(rgb).unsqueeze(0).requires_grad_(True)
        logits = model(x)
        probs = F.softmax(logits, dim=1)[0]
        k = int(probs.argmax()) if class_index is None else int(class_index)
        model.zero_grad()
        logits[0, k].backward()
    finally:
        for h in handles:
            h.remove()

    act, grad = saved["act"].detach(), saved["grad"].detach()  # (1, C, h, w)
    weights = grad.mean(dim=(2, 3), keepdim=True)               # mean gradient of each map
    cam = F.relu((weights * act).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=rgb.shape[:2], mode="bilinear", align_corners=False)[0, 0]
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    model.zero_grad()
    return cam.numpy().astype(np.float32), k, float(probs[k].detach())


def overlay(rgb: np.ndarray, heat: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """The photo with the heat map on top (jet: red = looked the most). Returns a new uint8 array."""
    colour = matplotlib.colormaps["jet"](np.clip(heat, 0, 1))[..., :3] * 255
    out = (1 - alpha) * rgb.astype(np.float32) + alpha * colour
    return np.clip(out, 0, 255).astype(np.uint8)

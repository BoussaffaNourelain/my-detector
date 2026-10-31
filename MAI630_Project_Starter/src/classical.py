"""Module 1: classical computer-vision operations as small functions.

Every function takes and returns NumPy arrays. Colour images are RGB (converted from OpenCV's BGR once, in
load_rgb). Greyscale images are 2-D uint8 arrays with values 0 (black) to 255 (white).
See docs/classical_spec.md.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


# ------------------------------------------------------------------ images and colour
def load_rgb(path: str | Path) -> np.ndarray:
    """Read an image file as an RGB array (height, width, 3)."""
    bgr = cv2.imread(str(path))
    if bgr is None:  # imread returns None instead of raising
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def to_grey(rgb: np.ndarray) -> np.ndarray:
    """Greyscale as the weighted sum 0.299 R + 0.587 G + 0.114 B."""
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def to_hsv(rgb: np.ndarray) -> np.ndarray:
    """Convert RGB to HSV (OpenCV hue runs 0-179)."""
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)


def brightness_contrast(img: np.ndarray, alpha: float = 1.0, beta: float = 0.0) -> np.ndarray:
    """alpha * pixel + beta, saturated to 0-255 (no uint8 wrap-around)."""
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)


# ------------------------------------------------------------------ filtering
def gaussian_blur(grey: np.ndarray, size: int = 5) -> np.ndarray:
    """Gaussian blur with a size x size kernel; size must be a positive odd number."""
    if size < 1 or size % 2 == 0:
        raise ValueError(f"size must be a positive odd number, got {size}")
    return cv2.GaussianBlur(grey, (size, size), 0)


def apply_kernel(grey: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Slide a hand-written kernel over the image (OpenCV filter2D); result clipped to 0-255."""
    return cv2.filter2D(grey, -1, kernel)


def convolve_by_hand(grey: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """The same operation as apply_kernel written with two loops; float32 output, not clipped.

    Like filter2D, the kernel is not flipped (correlation), and borders are reflected without repeating the
    edge pixel (BORDER_REFLECT_101 = np.pad mode "reflect").
    """
    kernel = np.asarray(kernel, dtype=np.float32)
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    padded = np.pad(grey.astype(np.float32), ((ph, ph), (pw, pw)), mode="reflect")
    out = np.zeros(grey.shape, dtype=np.float32)
    for y in range(grey.shape[0]):
        for x in range(grey.shape[1]):
            out[y, x] = (padded[y:y + kh, x:x + kw] * kernel).sum()
    return out


KERNELS = {
    "identity": np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float32),
    "average": np.full((3, 3), 1 / 9, dtype=np.float32),
    "sharpen": np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32),
    "horizontal_change": np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32),
}


# ------------------------------------------------------------------ thresholding
def threshold_fixed(grey: np.ndarray, t: float) -> np.ndarray:
    """Pixels strictly brighter than t become 255, the rest 0."""
    _, binary = cv2.threshold(grey, t, 255, cv2.THRESH_BINARY)
    return binary


def threshold_otsu(grey: np.ndarray) -> tuple[int, np.ndarray]:
    """Let Otsu's method choose the threshold; returns (t, binary)."""
    t, binary = cv2.threshold(grey, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return int(t), binary


def threshold_adaptive(grey: np.ndarray, block: int = 21, c: float = 10) -> np.ndarray:
    """One threshold per block x block neighbourhood: its Gaussian-weighted mean minus c."""
    if block < 3 or block % 2 == 0:
        raise ValueError(f"block must be an odd number >= 3, got {block}")
    return cv2.adaptiveThreshold(grey, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block, c)


# ------------------------------------------------------------------ edges
def sobel(grey: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Horizontal and vertical gradients (float, signed) and the magnitude scaled to 0-255 (uint8)."""
    gx = cv2.Sobel(grey, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(grey, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    peak = magnitude.max()
    if peak == 0:  # flat image: no gradient anywhere
        return gx, gy, np.zeros(grey.shape, dtype=np.uint8)
    return gx, gy, (255 * magnitude / peak).astype(np.uint8)


def laplacian(grey: np.ndarray) -> np.ndarray:
    """Absolute second derivative as uint8 (computed in float so negative values are not clipped first)."""
    return cv2.convertScaleAbs(cv2.Laplacian(grey, cv2.CV_64F))


def canny(grey: np.ndarray, low: float, high: float) -> np.ndarray:
    """Canny edge map (0 or 255); low must be smaller than high."""
    if low >= high:
        raise ValueError(f"low ({low}) must be smaller than high ({high})")
    return cv2.Canny(grey, low, high)


def edge_pixels(edges: np.ndarray) -> int:
    """Number of non-zero pixels in an edge map."""
    return int(np.count_nonzero(edges))

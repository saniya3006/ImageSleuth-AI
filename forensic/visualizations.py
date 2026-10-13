"""Exploratory image-consistency views.

These tools do not "detect" anything on their own. They re-render the image in
ways that make some inconsistencies easier for a human to notice:

* **Noise residual** - the image minus a median-filtered (denoised) copy. Camera
  sensor noise is usually fairly uniform; a pasted region from another source
  may have visibly different noise.
* **Local noise-level map** - the standard deviation of that residual per block,
  shown as a heat map.
* **Edge map (Canny)** - pasted objects sometimes have unnaturally sharp or
  unnaturally soft boundaries compared with the rest of the scene.
* **Colour channels** - RGB channels and the YCrCb channels used inside JPEG.

Every view is labelled as an indicator for visual inspection, not proof.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .utils import heatmap, normalize_to_uint8, to_gray


@dataclass
class ConsistencyResult:
    """Images produced by :func:`run_consistency_tools`."""

    noise_residual: np.ndarray
    noise_level_map: np.ndarray
    edges: np.ndarray
    channels: dict[str, np.ndarray] = field(default_factory=dict)
    parameters: dict[str, str] = field(default_factory=dict)


def noise_residual(rgb: np.ndarray, kernel: int = 3, amplification: float = 8.0) -> np.ndarray:
    """Return the amplified absolute difference between the image and a median-filtered copy."""
    if kernel not in (3, 5, 7):
        raise ValueError("kernel must be 3, 5 or 7")
    gray = to_gray(rgb)
    denoised = cv2.medianBlur(gray, kernel)
    residual = cv2.absdiff(gray, denoised).astype(np.float32)
    return np.clip(residual * amplification, 0, 255).astype(np.uint8)


def noise_level_map(rgb: np.ndarray, block: int = 16, kernel: int = 3) -> np.ndarray:
    """Heat map of the per-block standard deviation of the noise residual (RGB output)."""
    if block < 4:
        raise ValueError("block must be at least 4 pixels")
    gray = to_gray(rgb)
    residual = gray.astype(np.float32) - cv2.medianBlur(gray, kernel).astype(np.float32)
    height, width = gray.shape
    rows, cols = max(1, height // block), max(1, width // block)
    cropped = residual[: rows * block, : cols * block]
    if cropped.size == 0:
        cropped = residual
        rows, cols, block = 1, 1, max(height, width)
    blocks = cropped.reshape(rows, cropped.shape[0] // rows, cols, cropped.shape[1] // cols)
    std_map = blocks.std(axis=(1, 3))
    normalized = normalize_to_uint8(std_map)
    upscaled = cv2.resize(normalized, (width, height), interpolation=cv2.INTER_NEAREST)
    return heatmap(upscaled)


def edge_map(rgb: np.ndarray, sigma: float = 0.33) -> np.ndarray:
    """Canny edges with thresholds derived automatically from the median intensity."""
    gray = cv2.GaussianBlur(to_gray(rgb), (3, 3), 0)
    median = float(np.median(gray))
    lower = int(max(0, (1.0 - sigma) * median))
    upper = int(min(255, (1.0 + sigma) * median))
    if upper <= lower:
        upper = lower + 1
    return cv2.Canny(gray, lower, upper)


def channel_views(rgb: np.ndarray) -> dict[str, np.ndarray]:
    """Return individual R, G, B and Y, Cr, Cb channels as grayscale uint8 arrays."""
    ycrcb = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
    return {
        "Red": rgb[:, :, 0].copy(),
        "Green": rgb[:, :, 1].copy(),
        "Blue": rgb[:, :, 2].copy(),
        "Luma (Y)": ycrcb[:, :, 0].copy(),
        "Chroma red (Cr)": normalize_to_uint8(ycrcb[:, :, 1]),
        "Chroma blue (Cb)": normalize_to_uint8(ycrcb[:, :, 2]),
    }


def run_consistency_tools(
    rgb: np.ndarray, kernel: int = 3, amplification: float = 8.0, block: int = 16, sigma: float = 0.33
) -> ConsistencyResult:
    """Run all exploratory views with the given parameters."""
    return ConsistencyResult(
        noise_residual=noise_residual(rgb, kernel, amplification),
        noise_level_map=noise_level_map(rgb, block, kernel),
        edges=edge_map(rgb, sigma),
        channels=channel_views(rgb),
        parameters={
            "Median filter kernel": f"{kernel}x{kernel}",
            "Residual amplification": f"{amplification:g}x",
            "Noise-map block size": f"{block} px",
            "Canny threshold rule": f"median +/- {sigma:g} x median",
        },
    )


CONSISTENCY_LIMITATIONS = [
    "These are visual aids. Different noise or edge character between regions can come from focus, "
    "depth of field, lighting, motion blur, denoising or compression - not only from editing.",
    "Chroma (Cr/Cb) channels in JPEGs are typically subsampled, so blocky colour patterns are normal.",
]

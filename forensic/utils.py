"""Small shared helpers: identifiers, timestamps, and image/array conversions."""

from __future__ import annotations

import io
import secrets
from datetime import datetime, timezone

import cv2
import numpy as np
from PIL import Image


def new_investigation_id(now: datetime | None = None) -> str:
    """Return a unique, human-readable investigation identifier.

    Format: ``ISA-YYYYMMDD-HHMMSS-XXXXXXXX`` where the suffix is 32 bits of
    cryptographically secure randomness, so two uploads in the same second
    still receive different identifiers.
    """
    now = now or datetime.now(timezone.utc)
    return f"ISA-{now:%Y%m%d-%H%M%S}-{secrets.token_hex(4).upper()}"


def utc_now() -> datetime:
    """Return the current time as a timezone-aware UTC datetime."""
    return datetime.now(timezone.utc)


def format_timestamp(moment: datetime) -> str:
    """Format a datetime for display and reports, e.g. ``2026-10-13 18:04:05 UTC``."""
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def human_size(num_bytes: int) -> str:
    """Return a human-readable file size such as ``1.42 MB``."""
    size = float(num_bytes)
    for unit in ("bytes", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{int(size)} bytes" if unit == "bytes" else f"{size:.2f} {unit}"
        size /= 1024
    return f"{num_bytes} bytes"  # pragma: no cover - unreachable


def to_gray(rgb: np.ndarray) -> np.ndarray:
    """Convert an RGB uint8 array to a single-channel uint8 grayscale array."""
    if rgb.ndim == 2:
        return rgb
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def heatmap(gray: np.ndarray, colormap: int = cv2.COLORMAP_INFERNO) -> np.ndarray:
    """Apply a perceptual colour map to a uint8 grayscale array and return RGB."""
    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)
    bgr = cv2.applyColorMap(gray, colormap)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def normalize_to_uint8(values: np.ndarray) -> np.ndarray:
    """Linearly stretch an array to the 0-255 range (constant arrays become 0)."""
    values = values.astype(np.float32)
    low, high = float(values.min()), float(values.max())
    if high - low < 1e-9:
        return np.zeros(values.shape, dtype=np.uint8)
    return ((values - low) * (255.0 / (high - low))).astype(np.uint8)


def resize_max_dim(rgb: np.ndarray, max_dim: int) -> tuple[np.ndarray, float]:
    """Downscale so the longest side is at most ``max_dim``.

    Returns the (possibly unchanged) array and the scale factor that was applied
    (``1.0`` when no resizing was necessary).
    """
    height, width = rgb.shape[:2]
    longest = max(height, width)
    if longest <= max_dim:
        return rgb, 1.0
    scale = max_dim / float(longest)
    new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return cv2.resize(rgb, new_size, interpolation=cv2.INTER_AREA), scale


def array_to_png_bytes(array: np.ndarray, max_dim: int | None = None) -> bytes:
    """Encode an RGB or grayscale uint8 array as PNG bytes (optionally downscaled)."""
    if max_dim is not None:
        array, _ = resize_max_dim(array, max_dim)
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def array_to_jpeg_bytes(array: np.ndarray, max_dim: int | None = None, quality: int = 85) -> bytes:
    """Encode an RGB or grayscale uint8 array as JPEG bytes (for compact report illustrations)."""
    if max_dim is not None:
        array, _ = resize_max_dim(array, max_dim)
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="JPEG", quality=quality)
    return buffer.getvalue()

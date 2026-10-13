"""Shared pytest fixtures: synthetic images generated in memory (no files on disk)."""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image


def make_textured_rgb(width: int = 320, height: int = 240, seed: int = 0) -> np.ndarray:
    """Random but smooth-ish RGB content with plenty of texture."""
    rng = np.random.default_rng(seed)
    coarse = rng.integers(0, 256, (height // 8, width // 8, 3), dtype=np.uint8)
    image = np.array(Image.fromarray(coarse).resize((width, height), Image.BICUBIC))
    noise = rng.normal(0, 6, image.shape)
    return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def encode(rgb: np.ndarray, fmt: str = "JPEG", **kwargs) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(rgb).save(buffer, format=fmt, **kwargs)
    return buffer.getvalue()


def jpeg_with_exif(rgb: np.ndarray, software: str = "TestCam Firmware 1.0", gps: bool = False,
                   original: str = "2025:05:01 12:00:00", modified: str = "2025:05:01 12:00:00") -> bytes:
    exif = Image.Exif()
    exif[0x010F] = "TestMake"
    exif[0x0110] = "TestModel X"
    exif[0x0131] = software
    exif[0x0132] = modified
    exif[0x0112] = 1
    exif[0x8769] = {0x9003: original, 0x9004: original}  # sub-IFDs are written when assigned as dicts
    if gps:
        exif[0x8825] = {1: "N", 2: (48.0, 51.0, 29.52), 3: "E", 4: (2.0, 17.0, 40.2)}
    return encode(rgb, "JPEG", quality=92, exif=exif.tobytes())


@pytest.fixture
def textured_rgb() -> np.ndarray:
    return make_textured_rgb()


@pytest.fixture
def jpeg_bytes(textured_rgb) -> bytes:
    return encode(textured_rgb, "JPEG", quality=90)


@pytest.fixture
def png_bytes(textured_rgb) -> bytes:
    return encode(textured_rgb, "PNG")


@pytest.fixture
def exif_jpeg_bytes(textured_rgb) -> bytes:
    return jpeg_with_exif(textured_rgb)


@pytest.fixture
def gps_jpeg_bytes(textured_rgb) -> bytes:
    return jpeg_with_exif(textured_rgb, gps=True)

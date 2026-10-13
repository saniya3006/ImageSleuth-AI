"""Error Level Analysis (ELA).

How it works (plain English)
----------------------------
1. Take the decoded image and re-save it *in memory* as a JPEG at a known
   quality ``Q`` (default 90).
2. Subtract the re-saved pixels from the original pixels. JPEG compression
   is lossy, so every pixel changes a little - the "error level".
3. Amplify the small differences so they become visible.

Regions that have already been through JPEG compression at a similar quality
tend to change *less* when compressed again; content pasted in from elsewhere
(with a different compression history) may change by a noticeably different
amount. That difference *can* be a clue.

Why it is not proof
-------------------
* Error level depends strongly on content: edges, fine texture and noise are
  always brighter than smooth sky or walls.
* PNGs, screenshots, images resaved many times, resized images and images with
  mixed processing histories give ELA maps that are hard or impossible to read.
* Bright regions in an ELA map therefore NEVER prove tampering by themselves.

To reduce the "texture is always bright" problem, this module also computes a
tile-level *luma ELA-to-texture ratio* and highlights groups of neighbouring
tiles whose ratio is a strong statistical outlier compared with the rest of the
same image. That heuristic is uncalibrated and is reported strictly as an
indicator for human review.

ELA is most sensitive when the test quality is close to the quality at which
the *background* was previously compressed (the "JPEG ghost" effect), so a
quality sweep view is also provided as an exploratory aid.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field

import cv2
import numpy as np
import PIL
from PIL import Image

from .utils import heatmap

MIN_QUALITY = 50
MAX_QUALITY = 99
DEFAULT_QUALITY = 90
DEFAULT_TILE_SIZE = 32
DEFAULT_Z_THRESHOLD = 4.0

# Tiles with less texture than this (mean Sobel magnitude) are excluded from the
# ratio statistic: in flat areas both ELA and texture are near zero and the
# ratio is numerically unstable.
MIN_TILE_TEXTURE = 4.0
# Tiles whose mean error level is below this are never flagged (too faint to matter).
MIN_TILE_ERROR = 1.5


@dataclass
class ELAResult:
    """Output of :func:`perform_ela`."""

    quality: int
    amplification: float
    amplification_mode: str
    input_format: str
    diff_rgb: np.ndarray  # raw absolute difference, uint8 RGB
    amplified_rgb: np.ndarray  # visually amplified difference, uint8 RGB
    heatmap_rgb: np.ndarray  # colour-mapped luminance of the difference
    outlier_overlay_rgb: np.ndarray  # original with flagged tiles outlined
    stats: dict[str, float]
    tile_size: int
    z_threshold: float
    flagged_tiles: list[tuple[int, int, int, int]] = field(default_factory=list)  # (x, y, w, h)
    analysed_tiles: int = 0
    warnings: list[str] = field(default_factory=list)

    @property
    def is_jpeg_input(self) -> bool:
        return self.input_format == "JPEG"

    @property
    def flagged_fraction(self) -> float:
        return len(self.flagged_tiles) / self.analysed_tiles if self.analysed_tiles else 0.0

    @property
    def has_localized_outliers(self) -> bool:
        """True when a *small, localised* set of tiles stands out (JPEG inputs only).

        If a large share of the image is flagged, the pattern is global (e.g.
        heavy noise or mixed processing) rather than localised, so it is not
        treated as a localised indicator.
        """
        return self.is_jpeg_input and len(self.flagged_tiles) >= 2 and self.flagged_fraction <= 0.25

    def parameters(self) -> dict[str, str]:
        """Parameters needed to reproduce this run."""
        return {
            "Recompression format": "JPEG (in memory, never written to disk)",
            "JPEG quality": str(self.quality),
            "Chroma subsampling": "4:2:0",
            "Amplification": f"{self.amplification:.2f}x ({self.amplification_mode})",
            "Tile size": f"{self.tile_size} px",
            "Outlier statistic": "luma ELA / texture ratio per tile, robust z-score (median/MAD)",
            "Outlier threshold": f"z > {self.z_threshold:g} and >= 1 neighbouring tile also above threshold",
            "Encoder": f"Pillow {PIL.__version__} / libjpeg",
            "Input format": self.input_format,
        }


def recompress_jpeg(rgb: np.ndarray, quality: int) -> np.ndarray:
    """Re-encode an RGB array as JPEG at ``quality`` in memory and decode it again.

    Encoder settings are fixed (4:2:0 subsampling, no optimisation, baseline)
    so results are reproducible with the same Pillow/libjpeg versions.
    """
    buffer = io.BytesIO()
    Image.fromarray(rgb).save(buffer, format="JPEG", quality=int(quality), subsampling=2, optimize=False)
    buffer.seek(0)
    with Image.open(buffer) as recompressed:
        return np.array(recompressed.convert("RGB"), dtype=np.uint8)


def _robust_z(values: np.ndarray) -> np.ndarray:
    """Median/MAD based z-scores, resistant to the outliers we are looking for."""
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    scale = 1.4826 * mad if mad > 1e-9 else float(np.std(values)) or 1.0
    return (values - median) / scale


def _luma(rgb: np.ndarray) -> np.ndarray:
    """JPEG-style luma (Y of YCrCb) as float32."""
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)[:, :, 0].astype(np.float32)


def _tile_outliers(
    rgb: np.ndarray, recompressed: np.ndarray, tile: int, z_threshold: float
) -> tuple[list[tuple[int, int, int, int]], int]:
    """Find tiles whose luma ELA-to-texture ratio is an outlier within this image.

    * Luma (brightness) only: chroma is subsampled by JPEG, so colour edges
      produce large chroma errors that say little about compression history.
    * Ratio to texture: compares "how much did it change" with "how much detail
      is there", so ordinary busy texture is not flagged just for being busy.
    * Neighbour rule: an above-threshold tile is kept only if at least one of its
      eight neighbours is also above threshold. Inserted objects usually span
      several tiles, while isolated single tiles are mostly noise.
    """
    luma = _luma(rgb)
    error = np.abs(luma - _luma(recompressed))
    texture = cv2.magnitude(cv2.Sobel(luma, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(luma, cv2.CV_32F, 0, 1, ksize=3))

    height, width = luma.shape
    cells: list[tuple[int, int]] = []
    errors: list[float] = []
    textures: list[float] = []
    for row, y in enumerate(range(0, height - tile + 1, tile)):
        for col, x in enumerate(range(0, width - tile + 1, tile)):
            mean_texture = float(texture[y:y + tile, x:x + tile].mean())
            if mean_texture < MIN_TILE_TEXTURE:
                continue
            cells.append((row, col))
            errors.append(float(error[y:y + tile, x:x + tile].mean()))
            textures.append(mean_texture)

    if len(cells) < 8:  # too few textured tiles for meaningful statistics
        return [], len(cells)

    error_arr = np.array(errors)
    ratio = np.log((error_arr + 0.5) / (np.array(textures) + 1.0))
    z_scores = _robust_z(ratio)
    above = {cell for cell, z, err in zip(cells, z_scores, error_arr) if z > z_threshold and err >= MIN_TILE_ERROR}
    kept = [
        (row, col) for row, col in above
        if any((row + dr, col + dc) in above for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc)
    ]
    flagged = [(col * tile, row * tile, tile, tile) for row, col in sorted(kept)]
    return flagged, len(cells)


def quality_sweep(
    rgb: np.ndarray, qualities: tuple[int, ...] = (60, 65, 70, 75, 80, 85, 90, 95), window: int = 9
) -> list[tuple[int, np.ndarray, float]]:
    """Exploratory "JPEG ghost" sweep (simplified from Farid, 2009).

    For each quality the image is recompressed and the squared luma difference
    is averaged over a ``window`` x ``window`` neighbourhood, then normalised
    to 0-255 *within that quality*. If part of an image was previously
    compressed at quality q, that part tends to turn **dark** (changes little) at
    q while differently-compressed parts stay bright.

    Returns a list of ``(quality, heat-map RGB, mean luma error)``. This is a
    visual aid only; no indicator is derived from it.
    """
    luma = _luma(rgb)
    results: list[tuple[int, np.ndarray, float]] = []
    for quality in qualities:
        if not MIN_QUALITY <= quality <= MAX_QUALITY:
            raise ValueError(f"quality must be between {MIN_QUALITY} and {MAX_QUALITY}")
        difference = (luma - _luma(recompress_jpeg(rgb, quality))) ** 2
        smoothed = cv2.blur(difference, (window, window))
        low, high = float(smoothed.min()), float(smoothed.max())
        normalized = ((smoothed - low) / (high - low) * 255).astype(np.uint8) if high > low else np.zeros_like(
            smoothed, dtype=np.uint8)
        results.append((quality, heatmap(normalized), float(np.sqrt(difference.mean()))))
    return results


def perform_ela(
    rgb: np.ndarray,
    quality: int = DEFAULT_QUALITY,
    amplification: float | None = None,
    input_format: str = "JPEG",
    tile_size: int = DEFAULT_TILE_SIZE,
    z_threshold: float = DEFAULT_Z_THRESHOLD,
) -> ELAResult:
    """Run Error Level Analysis on an RGB ``uint8`` array.

    Args:
        rgb: decoded image (H x W x 3, uint8). It is not modified.
        quality: JPEG quality used for recompression (50-99).
        amplification: fixed multiplier for the difference image; ``None`` means
            automatic scaling so the 99.5th percentile maps to full brightness.
        input_format: the container format of the original upload ("JPEG"/"PNG").
        tile_size: side length in pixels of tiles used for outlier statistics.
        z_threshold: robust z-score above which a tile is flagged.

    Raises:
        ValueError: for invalid arguments.
    """
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("perform_ela expects an RGB uint8 array of shape (H, W, 3)")
    if not MIN_QUALITY <= int(quality) <= MAX_QUALITY:
        raise ValueError(f"quality must be between {MIN_QUALITY} and {MAX_QUALITY}")
    if amplification is not None and amplification <= 0:
        raise ValueError("amplification must be positive")
    if tile_size < 8:
        raise ValueError("tile_size must be at least 8 pixels")

    recompressed = recompress_jpeg(rgb, quality)
    diff = cv2.absdiff(rgb, recompressed)

    luminance = diff.max(axis=2)
    if amplification is None:
        reference = float(np.percentile(luminance, 99.5))
        amp = 255.0 / reference if reference > 0 else 1.0
        amp = float(min(max(amp, 1.0), 100.0))
        mode = "auto: 99.5th percentile scaled to full brightness"
    else:
        amp = float(amplification)
        mode = "fixed"
    amplified = np.clip(diff.astype(np.float32) * amp, 0, 255).astype(np.uint8)
    heat = heatmap(np.clip(luminance.astype(np.float32) * amp, 0, 255).astype(np.uint8))

    flagged, analysed = _tile_outliers(rgb, recompressed, tile_size, z_threshold)
    overlay = (rgb.astype(np.float32) * 0.55).astype(np.uint8)
    for x, y, w, h in flagged:
        cv2.rectangle(overlay, (x, y), (x + w - 1, y + h - 1), (255, 196, 0), 2)

    stats = {
        "mean_error": float(diff.mean()),
        "std_error": float(diff.std()),
        "max_error": float(diff.max()),
        "p99_error": float(np.percentile(luminance, 99)),
    }

    result = ELAResult(
        quality=int(quality),
        amplification=amp,
        amplification_mode=mode,
        input_format=input_format,
        diff_rgb=diff,
        amplified_rgb=amplified,
        heatmap_rgb=heat,
        outlier_overlay_rgb=overlay,
        stats=stats,
        tile_size=tile_size,
        z_threshold=z_threshold,
        flagged_tiles=flagged,
        analysed_tiles=analysed,
    )

    if input_format != "JPEG":
        result.warnings.append(
            f"The input is {input_format}, not JPEG. This run is a recompression EXPERIMENT only: a "
            "never-compressed (lossless) image changes everywhere when first saved as JPEG, so the map "
            "mainly reflects image content and no localised-outlier indicator is reported."
        )
    if stats["mean_error"] < 0.3:
        result.warnings.append(
            "Error levels are almost zero everywhere. The image may be extremely smooth or was saved at a "
            "quality very close to the test quality; ELA has little discriminating power here."
        )
    if analysed < 8:
        result.warnings.append(
            "Too few textured tiles for tile-level statistics; the outlier heuristic was not applied."
        )
    return result


ELA_LIMITATIONS = [
    "Bright areas show where recompression changed pixels most - typically edges, text and fine texture. "
    "Brightness alone never proves tampering.",
    "ELA is most interpretable for JPEGs saved once at a known quality. It is unreliable for PNGs, "
    "screenshots, heavily or repeatedly compressed images, resized images, and images with mixed "
    "processing histories.",
    "Sensitivity depends on the chosen quality: differences are clearest when the test quality is close to the "
    "quality at which part of the image was previously compressed. Use the quality sweep to explore this, and "
    "remember that trying many qualities also increases the chance of a coincidental pattern.",
    "Re-saving a manipulated image several times, adding noise, or resizing it can hide ELA differences.",
    "The tile-outlier heuristic compares tiles only against the same image and is uncalibrated: it has "
    "not been validated on a forensic benchmark, so it produces no probability or confidence score.",
]

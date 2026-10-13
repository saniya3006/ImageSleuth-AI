"""Upload validation and safe image decoding.

Uploaded files are treated as untrusted data:

* only JPEG and PNG are accepted, checked by *content* (magic bytes and the
  decoder), not just by file extension;
* file size and pixel count are bounded to prevent memory-exhaustion
  ("decompression bomb") attacks;
* files are decoded entirely in memory with Pillow - nothing from the upload is
  ever executed, written to disk, or passed to a shell.
"""

from __future__ import annotations

import io
import os
import warnings
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

DEFAULT_MAX_FILE_BYTES = 20 * 1024 * 1024  # 20 MB
MAX_IMAGE_PIXELS = 50_000_000  # 50 megapixels
MIN_DIMENSION = 16  # pixels; anything smaller cannot be meaningfully analysed

ALLOWED_EXTENSIONS = {".jpg": "JPEG", ".jpeg": "JPEG", ".png": "PNG"}
ALLOWED_FORMATS = {"JPEG", "PNG"}

JPEG_MAGIC = b"\xff\xd8\xff"
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"

# Make Pillow raise (instead of merely warn) for images above our pixel limit.
Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS


class ImageValidationError(ValueError):
    """Raised when an uploaded file is rejected. The message is safe to show to users."""


@dataclass(frozen=True)
class ValidatedImage:
    """Properties of an upload that passed validation."""

    filename: str
    size_bytes: int
    format: str
    width: int
    height: int
    mode: str
    warnings: tuple[str, ...] = field(default_factory=tuple)

    @property
    def megapixels(self) -> float:
        return self.width * self.height / 1_000_000


def sniff_format(data: bytes) -> str | None:
    """Identify the container format from its leading magic bytes."""
    if data.startswith(PNG_MAGIC):
        return "PNG"
    if data.startswith(JPEG_MAGIC):
        return "JPEG"
    return None


def safe_filename(filename: str) -> str:
    """Return only the base name, without any directory components supplied by the client."""
    base = os.path.basename(filename.replace("\\", "/")).strip()
    return base or "unnamed"


def validate_image_bytes(
    data: bytes,
    filename: str,
    max_bytes: int = DEFAULT_MAX_FILE_BYTES,
) -> ValidatedImage:
    """Validate an uploaded image and return its basic properties.

    Raises:
        ImageValidationError: with a user-friendly explanation if the file is
            empty, too large, of an unsupported type, corrupt or truncated.
    """
    name = safe_filename(filename)
    if not data:
        raise ImageValidationError("The uploaded file is empty.")
    if len(data) > max_bytes:
        raise ImageValidationError(
            f"The file is {len(data) / 1_048_576:.1f} MB; the maximum allowed size is "
            f"{max_bytes / 1_048_576:.0f} MB."
        )

    extension = os.path.splitext(name)[1].lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise ImageValidationError(
            f"Unsupported file extension '{extension or '(none)'}'. Please upload a .jpg, .jpeg or .png file."
        )

    sniffed = sniff_format(data)
    if sniffed is None:
        raise ImageValidationError(
            "The file content is not a JPEG or PNG image (the file signature does not match). "
            "It may be corrupt, or a different file type renamed with an image extension."
        )

    notes: list[str] = []
    if ALLOWED_EXTENSIONS[extension] != sniffed:
        notes.append(
            f"The extension '{extension}' does not match the actual content ({sniffed}). "
            f"The file was analysed as {sniffed}. A mismatched extension is common after renaming "
            "and is not by itself evidence of manipulation."
        )

    # Pass 1: structural verification without decoding pixel data.
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as probe:
                probe.verify()
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise ImageValidationError(
            f"The image dimensions exceed the safety limit of {MAX_IMAGE_PIXELS / 1e6:.0f} megapixels."
        ) from exc
    except UnidentifiedImageError as exc:
        raise ImageValidationError("The image could not be identified; the file appears to be corrupt.") from exc
    except Exception as exc:  # Pillow raises many exception types for malformed files
        raise ImageValidationError(f"The image structure is invalid or corrupt ({type(exc).__name__}).") from exc

    # Pass 2: full decode (verify() does not decode pixels, so truncation is caught here).
    try:
        with Image.open(io.BytesIO(data)) as image:
            image.load()
            width, height = image.size
            image_format = image.format or sniffed
            mode = image.mode
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise ImageValidationError(
            f"The image dimensions exceed the safety limit of {MAX_IMAGE_PIXELS / 1e6:.0f} megapixels."
        ) from exc
    except Exception as exc:
        raise ImageValidationError(
            f"The image data is truncated or corrupt and could not be decoded ({type(exc).__name__})."
        ) from exc

    if image_format not in ALLOWED_FORMATS:
        raise ImageValidationError(f"Decoded format '{image_format}' is not supported.")
    if width < MIN_DIMENSION or height < MIN_DIMENSION:
        raise ImageValidationError(
            f"The image is only {width}x{height} pixels; at least {MIN_DIMENSION}x{MIN_DIMENSION} is required."
        )

    return ValidatedImage(
        filename=name,
        size_bytes=len(data),
        format=image_format,
        width=width,
        height=height,
        mode=mode,
        warnings=tuple(notes),
    )


def decode_to_rgb(data: bytes) -> np.ndarray:
    """Decode image bytes into a new RGB ``uint8`` array for analysis.

    The pixel grid is kept in its *stored* orientation (EXIF orientation is not
    applied) so that JPEG 8x8 block positions remain aligned for ELA.
    Transparency is discarded and 16-bit images are reduced to 8 bits.
    """
    try:
        with Image.open(io.BytesIO(data)) as image:
            image.load()
            if image.mode in ("I;16", "I;16B", "I;16L", "I"):
                array = np.asarray(image, dtype=np.float64)
                scale = 255.0 / max(float(array.max()), 1.0)
                gray = np.clip(array * scale, 0, 255).astype(np.uint8)
                return np.stack([gray] * 3, axis=-1)
            if image.mode in ("RGBA", "LA", "PA") or (image.mode == "P" and "transparency" in image.info):
                image = image.convert("RGBA").convert("RGB")
            rgb = image.convert("RGB")
            return np.array(rgb, dtype=np.uint8)
    except Exception as exc:
        raise ImageValidationError(f"Unable to decode image for analysis ({type(exc).__name__}).") from exc


def decode_for_display(data: bytes) -> Image.Image:
    """Return an RGB Pillow image with EXIF orientation applied, for on-screen preview only."""
    try:
        with Image.open(io.BytesIO(data)) as image:
            image.load()
            oriented = ImageOps.exif_transpose(image)
            return oriented.convert("RGB")
    except Exception as exc:
        raise ImageValidationError(f"Unable to decode image for preview ({type(exc).__name__}).") from exc

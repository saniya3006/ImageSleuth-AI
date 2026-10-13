"""Software-behaviour tests for upload validation and safe decoding (incl. malformed input)."""

import numpy as np
import pytest

from forensic.case import open_case
from forensic.validation import (
    ImageValidationError,
    decode_for_display,
    decode_to_rgb,
    safe_filename,
    sniff_format,
    validate_image_bytes,
)
from tests.conftest import encode


def test_valid_jpeg(jpeg_bytes):
    info = validate_image_bytes(jpeg_bytes, "photo.jpg")
    assert (info.format, info.width, info.height, info.mode) == ("JPEG", 320, 240, "RGB")
    assert info.size_bytes == len(jpeg_bytes)
    assert info.warnings == ()


def test_valid_png(png_bytes):
    assert validate_image_bytes(png_bytes, "scan.PNG").format == "PNG"


def test_empty_file_rejected():
    with pytest.raises(ImageValidationError, match="empty"):
        validate_image_bytes(b"", "a.jpg")


def test_oversize_rejected(jpeg_bytes):
    with pytest.raises(ImageValidationError, match="maximum"):
        validate_image_bytes(jpeg_bytes, "a.jpg", max_bytes=100)


@pytest.mark.parametrize("name", ["image.gif", "image.exe", "image", "image.jpg.bat"])
def test_disallowed_extensions(jpeg_bytes, name):
    with pytest.raises(ImageValidationError, match="extension"):
        validate_image_bytes(jpeg_bytes, name)


def test_text_file_renamed_as_image_rejected():
    with pytest.raises(ImageValidationError, match="signature"):
        validate_image_bytes(b"hello, I am not an image", "fake.png")


def test_truncated_jpeg_rejected(jpeg_bytes):
    with pytest.raises(ImageValidationError):
        validate_image_bytes(jpeg_bytes[: len(jpeg_bytes) // 3], "cut.jpg")


def test_corrupted_png_rejected(png_bytes):
    corrupted = png_bytes[:40] + b"\x00" * 200 + png_bytes[240:]
    with pytest.raises(ImageValidationError):
        validate_image_bytes(corrupted, "bad.png")


def test_extension_mismatch_produces_warning(png_bytes):
    info = validate_image_bytes(png_bytes, "really_a_png.jpg")
    assert info.format == "PNG"
    assert any("does not match" in w for w in info.warnings)


def test_tiny_image_rejected():
    tiny = encode(np.zeros((4, 4, 3), np.uint8), "PNG")
    with pytest.raises(ImageValidationError, match="at least"):
        validate_image_bytes(tiny, "tiny.png")


def test_sniff_format():
    assert sniff_format(b"\xff\xd8\xff\xe0rest") == "JPEG"
    assert sniff_format(b"\x89PNG\r\n\x1a\nrest") == "PNG"
    assert sniff_format(b"GIF89a") is None


def test_safe_filename_strips_paths():
    assert safe_filename("../../etc/passwd.jpg") == "passwd.jpg"
    assert safe_filename("C:\\Users\\x\\photo.png") == "photo.png"
    assert safe_filename("") == "unnamed"


def test_decode_handles_rgba_and_grayscale():
    rgba = np.zeros((32, 32, 4), np.uint8)
    rgba[..., 3] = 128
    assert decode_to_rgb(encode(rgba, "PNG")).shape == (32, 32, 3)
    gray = np.full((32, 32), 77, np.uint8)
    decoded = decode_to_rgb(encode(gray, "PNG"))
    assert decoded.shape == (32, 32, 3) and decoded.dtype == np.uint8


def test_decode_corrupt_raises_validation_error():
    with pytest.raises(ImageValidationError):
        decode_to_rgb(b"\xff\xd8\xff garbage")
    with pytest.raises(ImageValidationError):
        decode_for_display(b"nope")


def test_open_case_hashes_first_and_reports_errors_gracefully():
    case = open_case(b"not an image", "evil.jpg")
    assert not case.is_valid
    assert case.validation_error
    assert len(case.sha256) == 64  # hash recorded even for rejected files


def test_open_case_valid(exif_jpeg_bytes):
    case = open_case(exif_jpeg_bytes, "dir/evidence.jpg")
    assert case.is_valid and case.filename == "evidence.jpg"
    assert case.investigation_id.startswith("ISA-")
    assert case.test_status["Metadata analysis"] == "completed"
    assert not case.rgb.flags.writeable  # analysis copy is read-only
    assert case.original_bytes == exif_jpeg_bytes

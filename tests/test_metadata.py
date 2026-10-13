"""Software-behaviour tests for metadata extraction, GPS handling and observations."""

import pytest

from forensic.metadata import (
    UNAVAILABLE,
    dms_to_decimal,
    extract_metadata,
    gps_display,
    redact_gps_fields,
)
from tests.conftest import jpeg_with_exif


def test_no_exif_is_observation_not_indicator(png_bytes):
    result = extract_metadata(png_bytes)
    assert result.has_exif is False
    assert result.camera_make == UNAVAILABLE
    assert result.indicators == []
    assert any("NOT evidence of manipulation" in o for o in result.observations)


def test_extracts_camera_fields(exif_jpeg_bytes):
    result = extract_metadata(exif_jpeg_bytes)
    assert result.has_exif
    assert result.camera_make == "TestMake"
    assert result.camera_model == "TestModel X"
    assert result.datetime_original == "2025:05:01 12:00:00"
    assert result.datetime_digitized == "2025:05:01 12:00:00"
    assert result.software == "TestCam Firmware 1.0"
    assert result.orientation.startswith("1")
    assert result.gps is None
    assert result.indicators == []  # camera firmware + consistent timestamps


def test_editing_software_and_timestamp_mismatch_are_indicators(textured_rgb):
    data = jpeg_with_exif(textured_rgb, software="Adobe Photoshop 25.0", modified="2025:06:01 09:00:00")
    result = extract_metadata(data)
    assert len(result.indicators) == 2
    assert any("Photoshop" in i for i in result.indicators)
    assert any("differs" in i for i in result.indicators)


def test_gps_decimal_conversion(gps_jpeg_bytes):
    result = extract_metadata(gps_jpeg_bytes)
    assert result.has_gps
    assert result.gps["latitude"] == pytest.approx(48.8582, abs=1e-4)
    assert result.gps["longitude"] == pytest.approx(2.2945, abs=1e-4)


def test_gps_redacted_by_default(gps_jpeg_bytes):
    result = extract_metadata(gps_jpeg_bytes)
    redacted = dict(result.summary_rows())["GPS"]
    assert "redacted" in redacted and "48." not in redacted
    shown = dict(result.summary_rows(include_gps=True))["GPS"]
    assert shown.startswith("48.858")
    assert gps_display(None, True) == UNAVAILABLE


def test_dms_southern_western_hemisphere_is_negative():
    assert dms_to_decimal((33.0, 52.0, 4.0), "S") == pytest.approx(-33.867778, abs=1e-5)
    assert dms_to_decimal((151.0, 12.0, 36.0), "W") == pytest.approx(-151.21, abs=1e-5)
    assert dms_to_decimal(("bad",), "N") is None


def test_redact_gps_fields():
    assert redact_gps_fields({"Make": "x", "GPSLatitude": "1"}) == {"Make": "x"}


def test_garbage_bytes_do_not_raise():
    result = extract_metadata(b"definitely not an image")
    assert result.error is not None
    assert result.has_exif is False

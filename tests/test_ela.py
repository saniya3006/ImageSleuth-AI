"""Software-behaviour tests for Error Level Analysis (shapes, determinism, parameters, warnings)."""

import numpy as np
import pytest

from forensic.ela import perform_ela, quality_sweep, recompress_jpeg
from forensic.validation import decode_to_rgb


def test_output_shapes_and_types(jpeg_bytes):
    rgb = decode_to_rgb(jpeg_bytes)
    result = perform_ela(rgb, quality=90)
    for image in (result.diff_rgb, result.amplified_rgb, result.heatmap_rgb, result.outlier_overlay_rgb):
        assert image.shape == rgb.shape and image.dtype == np.uint8
    assert result.quality == 90 and result.is_jpeg_input
    assert set(result.stats) == {"mean_error", "std_error", "max_error", "p99_error"}


def test_deterministic(jpeg_bytes):
    rgb = decode_to_rgb(jpeg_bytes)
    first, second = perform_ela(rgb, quality=85), perform_ela(rgb, quality=85)
    assert np.array_equal(first.diff_rgb, second.diff_rgb)
    assert first.flagged_tiles == second.flagged_tiles


def test_input_not_modified(jpeg_bytes):
    rgb = decode_to_rgb(jpeg_bytes)
    copy = rgb.copy()
    perform_ela(rgb)
    assert np.array_equal(rgb, copy)


def test_fixed_amplification_scales_difference(jpeg_bytes):
    rgb = decode_to_rgb(jpeg_bytes)
    result = perform_ela(rgb, amplification=10)
    expected = np.clip(result.diff_rgb.astype(np.float32) * 10, 0, 255).astype(np.uint8)
    assert np.array_equal(result.amplified_rgb, expected)
    assert result.amplification_mode == "fixed"


def test_uniform_image_has_low_error():
    flat = np.full((128, 128, 3), 120, np.uint8)
    result = perform_ela(flat)
    assert result.stats["mean_error"] < 1.0
    assert result.flagged_tiles == []
    assert any("almost zero" in w for w in result.warnings)


def test_png_input_is_labelled_experiment(png_bytes):
    result = perform_ela(decode_to_rgb(png_bytes), input_format="PNG")
    assert any("EXPERIMENT" in w for w in result.warnings)
    assert result.has_localized_outliers is False  # never an indicator for non-JPEG


def test_recompress_jpeg_is_lossy_but_close(textured_rgb):
    out = recompress_jpeg(textured_rgb, 95)
    assert out.shape == textured_rgb.shape
    assert 0 < np.abs(out.astype(int) - textured_rgb.astype(int)).mean() < 10


@pytest.mark.parametrize("kwargs", [{"quality": 10}, {"quality": 100}, {"amplification": 0}, {"tile_size": 4}])
def test_invalid_parameters(textured_rgb, kwargs):
    with pytest.raises(ValueError):
        perform_ela(textured_rgb, **kwargs)


def test_wrong_array_type():
    with pytest.raises(ValueError):
        perform_ela(np.zeros((10, 10), np.uint8))


def test_parameters_are_documented(jpeg_bytes):
    params = perform_ela(decode_to_rgb(jpeg_bytes), quality=88).parameters()
    assert params["JPEG quality"] == "88"
    assert "Chroma subsampling" in params and "Encoder" in params


def test_quality_sweep_outputs(textured_rgb):
    sweep = quality_sweep(textured_rgb, qualities=(70, 90))
    assert [q for q, _, _ in sweep] == [70, 90]
    for _, image, rms in sweep:
        assert image.shape == textured_rgb.shape and image.dtype == np.uint8 and rms >= 0
    with pytest.raises(ValueError):
        quality_sweep(textured_rgb, qualities=(20,))

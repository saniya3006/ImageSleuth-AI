"""Software-behaviour tests for the copy-move pipeline (structure, parameters, edge cases).

Whether the detector finds real forgeries is a forensic-accuracy question and
is covered separately in ``tests/synthetic_benchmarks``.
"""

import numpy as np
import pytest

from forensic.copy_move import (
    STATUS_INSUFFICIENT,
    CopyMoveParams,
    detect_copy_move,
)


def test_blank_image_reports_insufficient_features():
    result = detect_copy_move(np.full((200, 200, 3), 90, np.uint8))
    assert result.status == STATUS_INSUFFICIENT
    assert result.clusters == []
    assert "NOT a clean result" in result.status_text()
    assert result.visualization_rgb.shape == (200, 200, 3)


def test_textured_image_returns_structured_result(textured_rgb):
    result = detect_copy_move(textured_rgb)
    assert result.keypoints_detected > 0
    assert result.status in {"candidate_regions_found", "no_consistent_duplication", "insufficient_features"}
    assert result.visualization_rgb.shape == textured_rgb.shape


def test_input_not_modified(textured_rgb):
    copy = textured_rgb.copy()
    detect_copy_move(textured_rgb)
    assert np.array_equal(copy, textured_rgb)


def test_reproducible(textured_rgb):
    a = detect_copy_move(textured_rgb)
    b = detect_copy_move(textured_rgb)
    assert (a.status, a.candidate_matches, len(a.clusters)) == (b.status, b.candidate_matches, len(b.clusters))


def test_large_image_is_downscaled_and_coordinates_mapped_back():
    rng = np.random.default_rng(3)
    big = rng.integers(0, 255, (900, 2400, 3), dtype=np.uint8)
    result = detect_copy_move(big, CopyMoveParams(analysis_max_dim=800))
    assert result.analysis_scale == pytest.approx(800 / 2400)
    assert result.visualization_rgb.shape == big.shape


@pytest.mark.parametrize("field, value", [("ratio", 0.1), ("min_inliers", 2), ("max_hamming", 0),
                                          ("min_distance_px", 1), ("analysis_max_dim", 10)])
def test_invalid_parameters(textured_rgb, field, value):
    with pytest.raises(ValueError):
        detect_copy_move(textured_rgb, CopyMoveParams(**{field: value}))


def test_parameters_documented():
    shown = CopyMoveParams().as_display_dict()
    assert "Ratio test (g2NN)" in shown and "Geometric model" in shown

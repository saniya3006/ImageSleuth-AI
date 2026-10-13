"""Software-behaviour tests for consistency views and the rule-based summary."""

import numpy as np
import pytest

from forensic.integrity import compare_with_reference
from forensic.metadata import MetadataResult
from forensic.summary import (
    CORE_TESTS,
    INCOMPLETE,
    NO_INDICATORS,
    POTENTIAL_INDICATORS,
    TEST_COMPLETED,
    build_summary,
)
from forensic.utils import human_size, new_investigation_id
from forensic.visualizations import channel_views, edge_map, noise_level_map, noise_residual, run_consistency_tools


def test_consistency_outputs(textured_rgb):
    result = run_consistency_tools(textured_rgb)
    h, w = textured_rgb.shape[:2]
    assert result.noise_residual.shape == (h, w)
    assert result.noise_level_map.shape == (h, w, 3)
    assert result.edges.shape == (h, w) and set(np.unique(result.edges)) <= {0, 255}
    assert set(channel_views(textured_rgb)) == {"Red", "Green", "Blue", "Luma (Y)", "Chroma red (Cr)", "Chroma blue (Cb)"}
    assert result.parameters


def test_flat_image_has_zero_residual():
    flat = np.full((64, 64, 3), 50, np.uint8)
    assert noise_residual(flat).max() == 0
    assert edge_map(flat).max() == 0
    assert noise_level_map(flat).shape == (64, 64, 3)


def test_invalid_visualization_params(textured_rgb):
    with pytest.raises(ValueError):
        noise_residual(textured_rgb, kernel=4)
    with pytest.raises(ValueError):
        noise_level_map(textured_rgb, block=2)


def _all_done():
    return {name: TEST_COMPLETED for name in CORE_TESTS}


def test_summary_incomplete_when_not_run():
    summary = build_summary({}, None, None, None)
    assert summary.status == INCOMPLETE
    assert len(summary.incomplete) == len(CORE_TESTS)


def test_summary_no_indicators_wording():
    summary = build_summary(_all_done(), MetadataResult(has_exif=False), None, None)
    assert summary.status == NO_INDICATORS
    assert "does NOT establish" in summary.explanation


def test_summary_metadata_indicator():
    meta = MetadataResult(has_exif=True, indicators=["Software tag reads 'GIMP'"])
    assert build_summary(_all_done(), meta, None, None).status == POTENTIAL_INDICATORS


def test_summary_hash_mismatch_is_indicator():
    mismatch = compare_with_reference("a" * 64, "b" * 64)
    summary = build_summary(_all_done(), None, None, None, mismatch)
    assert summary.status == POTENTIAL_INDICATORS
    assert summary.indicators[0].source == "Integrity (SHA-256)"


def test_summary_never_contains_verdict_words():
    meta = MetadataResult(has_exif=True, indicators=["x"])
    for summary in (build_summary({}, None, None, None), build_summary(_all_done(), meta, None, None)):
        text = (summary.status + summary.explanation).lower()
        assert "fake" not in text and "%" not in text and "authentic image" not in text


def test_utils():
    ident = new_investigation_id()
    assert ident.startswith("ISA-") and len(ident) == len("ISA-20260101-000000-ABCDEF12")
    assert new_investigation_id() != new_investigation_id()
    assert human_size(512) == "512 bytes" and human_size(2048) == "2.00 KB"

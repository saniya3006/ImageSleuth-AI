"""SYNTHETIC sanity checks of forensic behaviour - NOT accuracy measurements.

These tests use the computer-generated demo images (with known, deliberately
made edits) to confirm that each detector responds to the kind of edit it is
designed for and stays quiet on the unedited version. Passing them shows the
pipeline behaves as intended on easy synthetic cases; it says NOTHING about
detection rates on real photographs, which would require a labelled
real-world benchmark (e.g. CASIA, Columbia, CoMoFoD) and proper evaluation.

Run only these:      pytest -m synthetic
Skip these:          pytest -m "not synthetic"
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest

from forensic.copy_move import detect_copy_move
from forensic.ela import perform_ela
from forensic.metadata import extract_metadata
from forensic.validation import ImageValidationError, decode_to_rgb, validate_image_bytes

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from generate_demo_images import generate  # noqa: E402

pytestmark = pytest.mark.synthetic


@pytest.fixture(scope="module")
def demo(tmp_path_factory):
    out = tmp_path_factory.mktemp("demo")
    manifest = generate(out)
    return out, manifest


def _rgb(path: Path):
    return decode_to_rgb(path.read_bytes())


def _overlaps(bbox_xyxy, box_xywh, margin=12) -> bool:
    x0, y0, x1, y1 = bbox_xyxy
    bx, by, bw, bh = box_xywh
    return x0 < bx + bw + margin and x1 > bx - margin and y0 < by + bh + margin and y1 > by - margin


def test_copy_move_found_on_synthetic_copy(demo):
    out, manifest = demo
    truth = manifest["files"]["02_synthetic_copy_move.jpg"]
    result = detect_copy_move(_rgb(out / "02_synthetic_copy_move.jpg"))
    assert result.found_candidates
    best = max(result.clusters, key=lambda c: c.inliers)
    boxes = {best.source_bbox, best.target_bbox}
    assert any(_overlaps(b, truth["source_box_xywh"]) for b in boxes)
    assert any(_overlaps(b, truth["target_box_xywh"]) for b in boxes)


def test_copy_move_quiet_on_unedited_synthetic_original(demo):
    out, _ = demo
    assert not detect_copy_move(_rgb(out / "01_synthetic_original.jpg")).found_candidates


def test_copy_move_flags_repetitive_pattern_known_false_positive(demo):
    """Documents a KNOWN limitation: repeating texture is matched although nothing was edited."""
    out, _ = demo
    assert detect_copy_move(_rgb(out / "05_repetitive_pattern_unedited.jpg")).found_candidates


def test_ela_outliers_fall_inside_spliced_region(demo):
    out, manifest = demo
    entry = manifest["files"]["03_synthetic_spliced.jpg"]
    truth = entry["pasted_box_xywh"]
    # ELA is run at the background's earlier quality, as the demo instructions explain.
    result = perform_ela(_rgb(out / "03_synthetic_spliced.jpg"), quality=entry["background_previous_quality"])
    assert result.has_localized_outliers
    inside = [t for t in result.flagged_tiles if _overlaps((t[0], t[1], t[0] + t[2], t[1] + t[3]), truth, margin=0)]
    assert len(inside) / len(result.flagged_tiles) >= 0.8


@pytest.mark.parametrize("quality", [75, 90])
def test_ela_quiet_on_unedited_synthetic_original(demo, quality):
    out, _ = demo
    assert not perform_ela(_rgb(out / "01_synthetic_original.jpg"), quality=quality).has_localized_outliers


def test_metadata_indicators_on_simulated_editor_tag(demo):
    out, _ = demo
    assert len(extract_metadata((out / "03_synthetic_spliced.jpg").read_bytes()).indicators) == 2
    assert extract_metadata((out / "01_synthetic_original.jpg").read_bytes()).indicators == []


def test_demo_corrupt_files_are_rejected(demo):
    out, _ = demo
    for name in ("06_corrupt_truncated.jpg", "07_not_an_image.png"):
        with pytest.raises(ImageValidationError):
            validate_image_bytes((out / name).read_bytes(), name)

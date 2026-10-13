"""Headless smoke tests of the Streamlit UI (no browser needed).

Uses ``streamlit.testing.v1.AppTest`` to execute ``app.py``, navigate every
page, press the main buttons and confirm no exception is raised. File upload
widgets cannot be driven by AppTest, so evidence is injected into session
state through the same ``open_case`` function the uploader uses.
"""

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from forensic.case import open_case  # noqa: E402

pytestmark = pytest.mark.app
APP = str(Path(__file__).resolve().parents[1] / "app.py")


def _app() -> AppTest:
    return AppTest.from_file(APP, default_timeout=120)


def _goto(at: AppTest, page_fragment: str) -> AppTest:
    radio = at.sidebar.radio[0]
    target = next(option for option in radio.options if page_fragment in option)
    return radio.set_value(target).run()


def _button(at: AppTest, label: str):
    return next(b for b in at.button if b.label == label)


def test_empty_state_renders_every_page():
    at = _app().run()
    assert not at.exception
    for page in ["Metadata", "Integrity", "Error Level", "Copy-Move", "Consistency", "Results", "About"]:
        at = _goto(at, page)
        assert not at.exception, page


def test_rejected_upload_shows_error():
    at = _app()
    at.session_state["case"] = open_case(b"not an image at all", "fake.jpg")
    at.run()
    assert not at.exception
    assert any("rejected" in md.value.lower() for md in at.markdown)


def test_full_workflow(jpeg_bytes):
    at = _app()
    at.session_state["case"] = open_case(jpeg_bytes, "demo.jpg")
    at.run()
    assert not at.exception

    _button(at, "Run all checks").click().run()
    assert not at.exception
    case = at.session_state["case"]
    assert case.ela is not None and case.copy_move is not None and case.consistency is not None

    for page in ["Metadata", "Integrity", "Error Level", "Copy-Move", "Consistency", "About"]:
        at = _goto(at, page)
        assert not at.exception, page

    at = _goto(at, "Integrity")
    at.text_input[0].set_value(case.sha256.upper())
    _button(at, "Verify").click().run()
    assert at.session_state["case"].hash_comparison.status == "match"

    at = _goto(at, "Error Level")
    _button(at, "Run ELA").click().run()
    assert not at.exception

    at = _goto(at, "Results")
    _button(at, "Generate PDF report").click().run()
    assert not at.exception
    assert at.session_state["report_pdf"].startswith(b"%PDF-")


def test_demo_loader_when_demo_assets_exist():
    at = _app().run()
    if not any(b.label == "Load demo image" for b in at.button):
        pytest.skip("assets/demo not generated")
    at.selectbox[0].set_value("02_synthetic_copy_move.jpg")
    _button(at, "Load demo image").click().run()
    assert not at.exception
    assert at.session_state["case"].filename == "02_synthetic_copy_move.jpg"


def test_copy_move_page_renders_cluster_table(tmp_path):
    """Regression: the candidate-region table (st.dataframe -> pyarrow) must render when clusters exist."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from generate_demo_images import generate

    generate(tmp_path)
    at = _app()
    at.session_state["case"] = open_case((tmp_path / "02_synthetic_copy_move.jpg").read_bytes(), "cm.jpg")
    at.run()
    at = _goto(at, "Copy-Move")
    _button(at, "Run copy-move analysis").click().run()
    assert not at.exception
    assert at.session_state["case"].copy_move.found_candidates
    assert len(at.dataframe) == 1

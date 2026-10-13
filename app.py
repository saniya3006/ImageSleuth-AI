"""ImageSleuth AI - Streamlit user interface.

Run from the project root with:  streamlit run app.py

This file only handles presentation and user interaction. All forensic logic
lives in the ``forensic`` package so it can be tested without a browser.
"""

from __future__ import annotations

import html
import logging
from pathlib import Path

import numpy as np
import streamlit as st

from forensic import APP_NAME, APP_TAGLINE, __version__
from forensic.case import EvidenceCase, open_case
from forensic.copy_move import COPY_MOVE_LIMITATIONS, CopyMoveParams, detect_copy_move
from forensic.ela import DEFAULT_QUALITY, ELA_LIMITATIONS, perform_ela, quality_sweep
from forensic.integrity import (
    STATUS_INVALID,
    STATUS_MATCH,
    STATUS_MISMATCH,
    compare_with_reference,
    verify_unchanged,
)
from forensic.metadata import gps_display, redact_gps_fields
from forensic.reporting import DISCLAIMER, EVIDENCE_HANDLING_NOTES, ReportOptions, generate_pdf_report, report_filename
from forensic.summary import (
    CORE_TESTS,
    INCOMPLETE,
    NO_INDICATORS,
    POTENTIAL_INDICATORS,
    TEST_COMPLETED,
    TEST_FAILED,
    build_summary,
)
from forensic.utils import format_timestamp, human_size
from forensic.validation import DEFAULT_MAX_FILE_BYTES, decode_for_display
from forensic.visualizations import CONSISTENCY_LIMITATIONS, run_consistency_tools

# Log events only - never image contents, bytes or metadata values.
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("imagesleuth")
DEMO_DIR = Path(__file__).resolve().parent / "assets" / "demo"

PAGES = [
    ":material/upload_file: Evidence Intake",
    ":material/badge: Metadata",
    ":material/fingerprint: Integrity (SHA-256)",
    ":material/layers: Error Level Analysis",
    ":material/content_copy: Copy-Move Analysis",
    ":material/tune: Consistency Tools",
    ":material/summarize: Results & Report",
    ":material/info: About & Limitations",
]
(PAGE_INTAKE, PAGE_METADATA, PAGE_INTEGRITY, PAGE_ELA, PAGE_COPY_MOVE, PAGE_CONSISTENCY, PAGE_RESULTS,
 PAGE_ABOUT) = PAGES

st.set_page_config(page_title=f"{APP_NAME} | Image Forensics", page_icon=":material/policy:", layout="wide",
                   initial_sidebar_state="expanded")

CSS = """
<style>
:root {
  --bg: #070B14; --panel: #0F1626; --panel-2: #131C30; --border: #1F2A40;
  --text: #E6EDF3; --muted: #9AA7BD; --accent: #2DD4BF; --accent-dim: rgba(45,212,191,.12);
  --warn: #FBBF24; --warn-dim: rgba(251,191,36,.10); --error: #FB7185; --error-dim: rgba(251,113,133,.10);
  --info: #7DD3FC; --info-dim: rgba(125,211,252,.10);
}
html, body, [class*="css"] { font-family: "Inter", "Segoe UI", system-ui, -apple-system, sans-serif; }
.stApp { background: radial-gradient(1200px 600px at 85% -10%, rgba(45,212,191,.07), transparent 60%), var(--bg); }
[data-testid="stSidebar"] { background: linear-gradient(180deg, #0A1120 0%, #070B14 100%); border-right: 1px solid var(--border); }
[data-testid="stSidebar"] [role="radiogroup"] label { padding: 6px 8px; border-radius: 8px; margin-bottom: 2px; }
[data-testid="stSidebar"] [role="radiogroup"] label:hover { background: var(--accent-dim); }
.block-container { padding-top: 2rem; max-width: 1280px; }
h1, h2, h3 { letter-spacing: -0.01em; }
.is-brand { display:flex; align-items:center; gap:10px; margin: 4px 0 2px 0; }
.is-brand .logo { width:36px; height:36px; border-radius:10px; display:grid; place-items:center;
  background: linear-gradient(135deg, #14B8A6, #0E7490); color:#04121A; font-weight:800; font-size:15px; }
.is-brand .name { font-weight:700; font-size:1.15rem; color: var(--text); }
.is-brand .ver { font-size:.72rem; color: var(--muted); }
.is-hero { border:1px solid var(--border); background: linear-gradient(135deg, #0F1A2E 0%, #0A111F 100%);
  border-radius:16px; padding: 22px 26px; margin-bottom: 18px; }
.is-hero .eyebrow { color: var(--accent); font-size:.78rem; font-weight:700; letter-spacing:.12em; text-transform:uppercase; }
.is-hero h1 { margin: 4px 0 6px 0; font-size: 1.75rem; color: var(--text); }
.is-hero p { color: var(--muted); margin: 0; max-width: 900px; }
.is-card { border:1px solid var(--border); background: var(--panel); border-radius:14px; padding:16px 18px; margin-bottom:12px; height: 100%; }
.is-card .label { color: var(--muted); font-size:.75rem; text-transform: uppercase; letter-spacing:.08em; font-weight:600; }
.is-card .value { color: var(--text); font-size:1.15rem; font-weight:650; margin-top:4px; word-break: break-word; }
.is-card .sub { color: var(--muted); font-size:.82rem; margin-top:2px; }
.is-callout { border-radius:12px; padding:12px 16px; margin: 6px 0 12px 0; border:1px solid; font-size:.92rem; line-height:1.5; }
.is-callout.ok { background: var(--accent-dim); border-color: rgba(45,212,191,.45); }
.is-callout.warn { background: var(--warn-dim); border-color: rgba(251,191,36,.45); }
.is-callout.error { background: var(--error-dim); border-color: rgba(251,113,133,.5); }
.is-callout.info { background: var(--info-dim); border-color: rgba(125,211,252,.4); }
.is-callout b.title { display:block; margin-bottom:2px; }
.is-pill { display:inline-block; padding: 2px 10px; border-radius: 999px; font-size: .75rem; font-weight: 700; border:1px solid; }
.is-pill.ok { color: var(--accent); border-color: rgba(45,212,191,.5); background: var(--accent-dim); }
.is-pill.warn { color: var(--warn); border-color: rgba(251,191,36,.5); background: var(--warn-dim); }
.is-pill.error { color: var(--error); border-color: rgba(251,113,133,.5); background: var(--error-dim); }
.is-pill.muted { color: var(--muted); border-color: var(--border); background: transparent; }
.is-kv { width:100%; border-collapse: collapse; font-size:.9rem; }
.is-kv td { padding: 7px 10px; border-bottom: 1px solid var(--border); vertical-align: top; }
.is-kv td:first-child { color: var(--muted); width: 38%; }
.is-kv td.mono, .is-mono { font-family: "JetBrains Mono", "Cascadia Code", Consolas, monospace; font-size:.85rem; word-break: break-all; }
.is-section-sub { color: var(--muted); margin-top:-6px; margin-bottom: 14px; }
.is-step { display:flex; gap:12px; align-items:flex-start; margin: 8px 0; color: var(--text); }
.is-step .n { flex: 0 0 26px; height:26px; border-radius:50%; display:grid; place-items:center; background: var(--accent-dim);
  color: var(--accent); font-weight:700; font-size:.85rem; border:1px solid rgba(45,212,191,.4); }
.is-side-case { border:1px solid var(--border); border-radius:12px; padding:10px 12px; background: var(--panel); font-size:.8rem; }
.is-side-case .k { color: var(--muted); }
.stButton > button[kind="primary"], .stDownloadButton > button[kind="primary"] { color: #04121A; font-weight: 700; }
[data-testid="stMetric"] { background: var(--panel); border:1px solid var(--border); border-radius:12px; padding: 10px 14px; }
[data-testid="stExpander"] { border:1px solid var(--border); border-radius: 12px; }
@media (max-width: 640px) { .is-hero { padding: 16px; } .is-hero h1 { font-size: 1.35rem; } }
</style>
"""


# --------------------------------------------------------------------------- UI helpers
def esc(value: object) -> str:
    """HTML-escape any value before it is placed in custom HTML (filenames and EXIF are untrusted)."""
    return html.escape(str(value))


def hero(eyebrow: str, title: str, text: str) -> None:
    st.markdown(f'<div class="is-hero"><div class="eyebrow">{esc(eyebrow)}</div><h1>{esc(title)}</h1>'
                f'<p>{esc(text)}</p></div>', unsafe_allow_html=True)


def callout(kind: str, title: str, text: str) -> None:
    st.markdown(f'<div class="is-callout {kind}"><b class="title">{esc(title)}</b>{esc(text)}</div>',
                unsafe_allow_html=True)


def card(label: str, value: str, sub: str = "") -> None:
    sub_html = f'<div class="sub">{esc(sub)}</div>' if sub else ""
    st.markdown(f'<div class="is-card"><div class="label">{esc(label)}</div><div class="value">{esc(value)}</div>'
                f'{sub_html}</div>', unsafe_allow_html=True)


def pill(text: str, kind: str) -> str:
    return f'<span class="is-pill {kind}">{esc(text)}</span>'


def kv_table(rows: list[tuple[str, str]], mono_keys: tuple[str, ...] = ()) -> None:
    body = "".join(
        f'<tr><td>{esc(k)}</td><td class="{"mono" if k in mono_keys else ""}">{esc(v)}</td></tr>' for k, v in rows
    )
    st.markdown(f'<table class="is-kv">{body}</table>', unsafe_allow_html=True)


def bullet_list(items: list[str]) -> None:
    st.markdown("\n".join(f"- {esc(item)}" for item in items))


def limitations(items: list[str], title: str = "Limitations - read before interpreting") -> None:
    with st.expander(title, icon=":material/warning:"):
        bullet_list(items)


def goto(page: str) -> None:
    st.session_state["nav"] = page


def next_button(page: str, label: str) -> None:
    st.button(label, on_click=goto, args=(page,), icon=":material/arrow_forward:")


def get_case() -> EvidenceCase | None:
    return st.session_state.get("case")


def require_case() -> EvidenceCase | None:
    """Return the current valid case, or render a helpful empty state."""
    case = get_case()
    if case is None:
        callout("info", "No evidence loaded",
                "Upload a JPEG or PNG image on the Evidence Intake page to begin. Nothing is analysed until you do.")
        st.button("Go to Evidence Intake", on_click=goto, args=(PAGE_INTAKE,), type="primary",
                  icon=":material/upload_file:")
        return None
    if not case.is_valid:
        callout("error", "The current file was rejected", case.validation_error or "Unknown validation error.")
        st.button("Upload a different file", on_click=goto, args=(PAGE_INTAKE,), icon=":material/upload_file:")
        return None
    return case


def invalidate_report() -> None:
    st.session_state.pop("report_pdf", None)


# --------------------------------------------------------------------------- analysis runners
def run_ela(case: EvidenceCase, quality: int, amplification: float | None, tile: int, z: float) -> None:
    try:
        case.ela = perform_ela(case.rgb, quality=quality, amplification=amplification,
                               input_format=case.validated.format, tile_size=tile, z_threshold=z)
        case.mark("Error Level Analysis", TEST_COMPLETED)
    except Exception as exc:
        case.ela = None
        case.mark("Error Level Analysis", TEST_FAILED, f"{type(exc).__name__}: {exc}")
        log.warning("ELA failed: %s", type(exc).__name__)
    invalidate_report()


def run_copy_move(case: EvidenceCase, params: CopyMoveParams) -> None:
    try:
        case.copy_move = detect_copy_move(case.rgb, params)
        case.mark("Copy-move analysis", TEST_COMPLETED)
    except Exception as exc:
        case.copy_move = None
        case.mark("Copy-move analysis", TEST_FAILED, f"{type(exc).__name__}: {exc}")
        log.warning("Copy-move failed: %s", type(exc).__name__)
    invalidate_report()


def run_consistency(case: EvidenceCase, kernel: int, amplification: float, block: int, sigma: float) -> None:
    try:
        case.consistency = run_consistency_tools(case.rgb, kernel, amplification, block, sigma)
        case.mark("Consistency tools", TEST_COMPLETED)
    except Exception as exc:
        case.consistency = None
        case.mark("Consistency tools", TEST_FAILED, f"{type(exc).__name__}: {exc}")
        log.warning("Consistency tools failed: %s", type(exc).__name__)
    invalidate_report()


def run_all(case: EvidenceCase) -> None:
    """Run every analysis with default parameters, showing progress."""
    steps = [
        ("Error Level Analysis", lambda: run_ela(case, DEFAULT_QUALITY, None, 32, 4.0)),
        ("Copy-move analysis", lambda: run_copy_move(case, CopyMoveParams())),
        ("Consistency tools", lambda: run_consistency(case, 3, 8.0, 16, 0.33)),
    ]
    progress = st.progress(0.0, text="Starting analysis...")
    with st.status("Running forensic checks", expanded=True) as status:
        st.write(f"SHA-256 already recorded at intake: `{case.sha256[:16]}...`")
        st.write("Metadata extracted at intake.")
        for index, (name, step) in enumerate(steps):
            progress.progress(index / len(steps), text=f"Running {name}...")
            step()
            outcome = case.test_status.get(name)
            st.write(f"{name}: **{outcome}**" + (f" - {case.errors[name]}" if name in case.errors else ""))
        progress.progress(1.0, text="All checks finished.")
        status.update(label="Forensic checks finished", state="complete", expanded=False)
    log.info("Full analysis run for %s", case.investigation_id)


# --------------------------------------------------------------------------- sidebar
def render_sidebar() -> str:
    with st.sidebar:
        st.markdown(f'<div class="is-brand"><div class="logo">IS</div><div><div class="name">{esc(APP_NAME)}</div>'
                    f'<div class="ver">v{esc(__version__)} - educational forensics</div></div></div>',
                    unsafe_allow_html=True)
        st.divider()
        page = st.radio("Navigation", PAGES, key="nav", label_visibility="collapsed")
        st.divider()
        case = get_case()
        if case is not None and case.is_valid:
            done = sum(1 for t in CORE_TESTS if case.test_status.get(t) == TEST_COMPLETED)
            st.markdown(
                f'<div class="is-side-case"><div class="k">Investigation</div><div class="is-mono">{esc(case.investigation_id)}</div>'
                f'<div class="k" style="margin-top:6px">File</div><div>{esc(case.filename)}</div></div>',
                unsafe_allow_html=True)
            st.progress(done / len(CORE_TESTS), text=f"Core checks: {done}/{len(CORE_TESTS)} complete")
        elif case is not None:
            st.caption("Current upload was rejected.")
        else:
            st.caption("No evidence loaded.")
        st.caption(":material/lock: Runs locally. Images are processed in memory and never sent to external services.")
    return page


# --------------------------------------------------------------------------- pages
def page_intake() -> None:
    hero("Step 1 - Evidence intake", "Upload an image for analysis",
         "Supported: JPEG / JPG / PNG up to 20 MB. The SHA-256 hash of the exact uploaded bytes is recorded "
         "before any analysis, and the original bytes are never modified.")

    uploader_key = f"uploader_{st.session_state.get('uploader_generation', 0)}"
    uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"], key=uploader_key,
                                help="Files are validated by content (magic bytes + decoder), not just extension.")
    if uploaded is not None:
        token = (uploaded.name, uploaded.size, uploaded.file_id)
        if st.session_state.get("upload_token") != token:
            with st.spinner("Hashing and validating..."):
                st.session_state["case"] = open_case(uploaded.getvalue(), uploaded.name, DEFAULT_MAX_FILE_BYTES)
            st.session_state["upload_token"] = token
            invalidate_report()
            log.info("New evidence received: %s", st.session_state["case"].investigation_id)

    demo_files = sorted(p for p in DEMO_DIR.glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if demo_files:
        with st.expander("Or load a SYNTHETIC demo image (generated by scripts/generate_demo_images.py)",
                         icon=":material/science:"):
            choice = st.selectbox("Demo image", [p.name for p in demo_files],
                                  help="Computer-generated samples with known edits - see assets/demo/manifest.json.")
            if st.button("Load demo image", icon=":material/science:"):
                path = DEMO_DIR / choice
                st.session_state["case"] = open_case(path.read_bytes(), path.name, DEFAULT_MAX_FILE_BYTES)
                st.session_state["upload_token"] = ("demo", choice)
                invalidate_report()

    case = get_case()
    if case is None:
        st.markdown("#### How a typical investigation works")
        for n, text in enumerate([
            "Upload a JPEG or PNG. Its SHA-256 fingerprint is recorded immediately.",
            "Review file properties and metadata (missing metadata is normal and not proof of anything).",
            "Run Error Level Analysis and copy-move analysis - or use 'Run all checks'.",
            "Inspect the visual evidence, parameters and limitations on each page.",
            "Generate a PDF report that separates observations from conclusions.",
        ], start=1):
            st.markdown(f'<div class="is-step"><div class="n">{n}</div><div>{esc(text)}</div></div>',
                        unsafe_allow_html=True)
        callout("info", "Need test images?",
                "Run  python scripts/generate_demo_images.py  to create clearly labelled synthetic samples in "
                "assets/demo/.")
        return

    if not case.is_valid:
        callout("error", "File rejected", case.validation_error or "The file could not be validated.")
        kv_table([("Filename", case.filename), ("Size", human_size(len(case.original_bytes))),
                  ("SHA-256 of rejected bytes", case.sha256)], mono_keys=("SHA-256 of rejected bytes",))
        st.caption("No analysis was performed. Please upload a valid JPEG or PNG image.")
        if st.button("Clear", icon=":material/delete:"):
            clear_case()
        return

    info = case.validated
    for warning in info.warnings:
        callout("warn", "Note", warning)

    left, right = st.columns([3, 2], gap="large")
    with left:
        st.image(decode_for_display(case.original_bytes), caption="Preview (EXIF orientation applied for display only)",
                 use_container_width=True)
    with right:
        st.markdown("#### File information")
        kv_table([
            ("Investigation ID", case.investigation_id),
            ("Received", format_timestamp(case.received_at)),
            ("Filename", case.filename),
            ("Format (by content)", info.format),
            ("Dimensions", f"{info.width} x {info.height} px"),
            ("Megapixels", f"{info.megapixels:.2f} MP"),
            ("File size", f"{human_size(info.size_bytes)} ({info.size_bytes:,} bytes)"),
            ("Colour mode", info.mode),
            ("SHA-256", case.sha256),
        ], mono_keys=("Investigation ID", "SHA-256"))
        st.write("")
        if st.button("Run all checks", type="primary", icon=":material/play_arrow:", use_container_width=True):
            run_all(case)
            callout("ok", "Done", "All checks finished. Open 'Results & Report' for the summary and PDF.")
        st.button("Open Results & Report", on_click=goto, args=(PAGE_RESULTS,), use_container_width=True,
                  icon=":material/summarize:")
        if st.button("Clear evidence and start over", use_container_width=True, icon=":material/delete:"):
            clear_case()


def clear_case() -> None:
    for key in ("case", "upload_token", "report_pdf"):
        st.session_state.pop(key, None)
    st.session_state["uploader_generation"] = st.session_state.get("uploader_generation", 0) + 1
    st.rerun()


def page_metadata() -> None:
    hero("Metadata", "Embedded metadata (EXIF)",
         "Camera, timestamp and software information stored inside the file. Metadata is easy to edit or strip, "
         "so it can support or question a story but never proves authenticity or manipulation.")
    case = require_case()
    if case is None:
        return
    meta = case.metadata
    if meta is None:
        callout("error", "Metadata unavailable", case.errors.get("Metadata analysis", "Metadata was not extracted."))
        return
    if meta.error:
        callout("warn", "Partial result", meta.error)

    show_gps = st.toggle("Show GPS location on screen", value=False,
                         help="GPS is hidden by default for privacy. This does not affect the PDF report setting.")
    col1, col2 = st.columns([1, 1], gap="large")
    with col1:
        st.markdown("#### Key fields")
        kv_table(meta.summary_rows(include_gps=show_gps))
    with col2:
        st.markdown("#### Indicators")
        if meta.indicators:
            for text in meta.indicators:
                callout("warn", "Indicator (not proof)", text)
        else:
            callout("ok", "No metadata indicators", "No editing-software tag or timestamp inconsistency was found.")
        st.markdown("#### Observations")
        for text in meta.observations or ["No additional observations."]:
            callout("info", "Observation", text)

    if meta.has_gps and show_gps and meta.gps and meta.gps.get("latitude") is not None:
        st.caption(f"Decimal coordinates: {gps_display(meta.gps, True)} (shown at your request).")

    with st.expander(f"All EXIF fields ({len(meta.fields)})", icon=":material/list:"):
        fields = meta.fields if show_gps else redact_gps_fields(meta.fields)
        if fields:
            kv_table(sorted(fields.items()))
        else:
            st.caption("No EXIF fields present.")
        if meta.gps and show_gps:
            st.markdown("**GPS IFD**")
            kv_table(sorted(meta.gps.get("raw", {}).items()))
    with st.expander("Container information (PNG chunks, JFIF, ICC, XMP)", icon=":material/inventory_2:"):
        if meta.container_info:
            kv_table(sorted(meta.container_info.items()))
        else:
            st.caption("No additional container information.")
    limitations([
        "Absent EXIF is normal for screenshots, PNG exports, messaging and social-media images.",
        "Any EXIF field, including camera model and timestamps, can be edited with free tools.",
        "A Software tag shows processing, which is often harmless (resizing, colour correction).",
    ])
    next_button(PAGE_INTEGRITY, "Next: Integrity")


def page_integrity() -> None:
    hero("Integrity", "SHA-256 integrity verification",
         "A SHA-256 hash is a 64-character fingerprint of the exact bytes. Changing even one bit changes the hash. "
         "It proves byte-level equality only when compared against a trusted reference hash.")
    case = require_case()
    if case is None:
        return
    st.markdown("#### SHA-256 of the uploaded bytes (computed at intake, before analysis)")
    st.code(case.sha256, language=None)
    unchanged = verify_unchanged(case.original_bytes, case.sha256)
    st.markdown(pill("Re-verified now: bytes unchanged" if unchanged else "Bytes changed!", "ok" if unchanged else "error"),
                unsafe_allow_html=True)

    st.markdown("#### Compare with a reference hash")
    with st.form("hash_form"):
        reference = st.text_input("Trusted reference SHA-256 (optional)", placeholder="64 hexadecimal characters",
                                  value=(case.hash_comparison.reference or "") if case.hash_comparison else "")
        submitted = st.form_submit_button("Verify", type="primary", icon=":material/verified:")
    if submitted:
        case.hash_comparison = compare_with_reference(case.sha256, reference)
        invalidate_report()
    comparison = case.hash_comparison
    if comparison is not None:
        kind = {STATUS_MATCH: "ok", STATUS_MISMATCH: "warn", STATUS_INVALID: "error"}.get(comparison.status, "info")
        callout(kind, comparison.status.replace("_", " ").upper(), comparison.message)
    callout("info", "What a hash can and cannot tell you",
            "Use a reference hash recorded when the image was first acquired (e.g. by the camera owner or an "
            "evidence log). A match shows the file is byte-identical to that original. A hash says nothing about "
            "whether the content was edited before the reference was recorded.")
    next_button(PAGE_ELA, "Next: Error Level Analysis")


def page_ela() -> None:
    hero("Error Level Analysis", "JPEG recompression comparison",
         "The image is re-saved in memory as a JPEG at a known quality and compared with itself. Areas with a "
         "different compression history may respond differently - but edges and texture are always brighter.")
    case = require_case()
    if case is None:
        return
    if case.validated.format != "JPEG":
        callout("warn", "Non-JPEG input - experiment only",
                f"This file is {case.validated.format}. ELA is designed for JPEGs; for lossless images it runs as a "
                "clearly labelled recompression experiment and no ELA indicator is produced.")

    with st.form("ela_form"):
        c1, c2, c3 = st.columns(3)
        quality = c1.slider("Recompression JPEG quality", 50, 99, DEFAULT_QUALITY,
                            help="Common choices are 90 or 95. Use the same value to reproduce a result.")
        auto = c2.radio("Amplification", ["Automatic", "Fixed"], horizontal=True) == "Automatic"
        fixed = c3.slider("Fixed amplification factor", 1, 60, 20,
                          help="Used only when Amplification = Fixed.")
        with st.expander("Advanced: tile outlier heuristic"):
            a1, a2 = st.columns(2)
            tile = a1.select_slider("Tile size (px)", options=[16, 32, 64], value=32)
            z = a2.slider("Robust z-score threshold", 2.0, 8.0, 4.0, 0.5)
        run = st.form_submit_button("Run ELA", type="primary", icon=":material/play_arrow:")
    if run:
        with st.spinner("Recompressing and comparing..."):
            run_ela(case, quality, None if auto else float(fixed), tile, z)

    if case.test_status.get("Error Level Analysis") == TEST_FAILED:
        callout("error", "ELA failed", case.errors.get("Error Level Analysis", "Unknown error"))
    ela = case.ela
    if ela is None:
        callout("info", "Not run yet", "Choose parameters and press Run ELA (defaults are fine for a first look).")
        limitations(ELA_LIMITATIONS)
        return

    for warning in ela.warnings:
        callout("warn", "Caution", warning)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Quality used", ela.quality)
    m2.metric("Mean error", f"{ela.stats['mean_error']:.2f}")
    m3.metric("99th percentile", f"{ela.stats['p99_error']:.0f}")
    m4.metric("Outlier tiles", f"{len(ela.flagged_tiles)} / {ela.analysed_tiles}")

    tab1, tab2, tab3 = st.tabs(["Original vs ELA", "Heat map", "Outlier tiles"])
    with tab1:
        a, b = st.columns(2)
        a.image(case.rgb, caption="Analysed image (stored orientation)", use_container_width=True)
        b.image(ela.amplified_rgb, caption=f"ELA difference amplified {ela.amplification:.1f}x", use_container_width=True)
    with tab2:
        st.image(ela.heatmap_rgb, caption="Error level heat map (brighter = larger recompression change)",
                 use_container_width=True)
    with tab3:
        st.image(ela.outlier_overlay_rgb, caption="Tiles outlined in yellow have an unusually high error-to-texture ratio",
                 use_container_width=True)
        if ela.has_localized_outliers:
            callout("warn", "Indicator (not proof)",
                    f"{len(ela.flagged_tiles)} tiles stand out relative to the rest of this image. Compare them with "
                    "the original: do they align with an object boundary that could have been inserted?")
        elif ela.flagged_tiles:
            callout("info", "Observation",
                    "Some tiles exceeded the threshold, but the pattern is not a small localised group (or the input "
                    "is not JPEG), so it is not reported as an indicator.")
        else:
            callout("ok", "No outlier tiles", "No tile exceeded the outlier threshold. This is not proof of authenticity.")

    with st.expander("How to read this", icon=":material/school:"):
        st.markdown(
            "- Uniform noise-like brightness across similar textures suggests a consistent compression history.\n"
            "- A region that is much brighter or darker **than similar texture elsewhere** may deserve a closer look.\n"
            "- Edges, text and high-contrast detail are naturally bright - compare like with like.\n"
            "- ELA is most sensitive when the test quality is close to the quality at which the *background* was "
            "previously saved. If you do not know it, use the quality sweep below.")

    with st.expander("Quality sweep - exploratory 'JPEG ghost' view", icon=":material/grid_view:"):
        st.caption("Recompresses at several qualities. A region that turns dark at one quality while the rest stays "
                   "bright may have been compressed at that quality before. Visual aid only - no indicator is derived, "
                   "and looking at many qualities increases the chance of coincidental patterns.")
        sweep_key = f"sweep_{case.investigation_id}"
        if st.button("Run quality sweep (60-95)", icon=":material/grid_view:"):
            with st.spinner("Recompressing at 8 quality levels..."):
                st.session_state[sweep_key] = quality_sweep(case.rgb)
        sweep = st.session_state.get(sweep_key)
        if sweep:
            for row in range(0, len(sweep), 4):
                cols = st.columns(4)
                for col, (q, image, rms) in zip(cols, sweep[row:row + 4]):
                    col.image(image, caption=f"Quality {q} - RMS luma change {rms:.2f}", use_container_width=True)
    st.markdown("#### Parameters (for reproducibility)")
    kv_table(list(ela.parameters().items()))
    limitations(ELA_LIMITATIONS)
    next_button(PAGE_COPY_MOVE, "Next: Copy-Move Analysis")


def page_copy_move() -> None:
    hero("Copy-move analysis", "Duplicated-region search with ORB keypoints",
         "Distinctive points are detected and matched against the rest of the same image. Matches are kept only if "
         "they are unambiguous, far apart, and many of them agree on one geometric transformation (RANSAC).")
    case = require_case()
    if case is None:
        return
    defaults = CopyMoveParams()
    with st.form("cm_form"):
        c1, c2, c3 = st.columns(3)
        ratio = c1.slider("Ratio test threshold", 0.50, 0.90, defaults.ratio, 0.05,
                          help="Lower = stricter: the best match must be clearly better than the next one.")
        min_dist = c2.slider("Min distance between matches (px)", 10, 200, int(defaults.min_distance_px), 5)
        min_inliers = c3.slider("Min consistent matches per region", 4, 40, defaults.min_inliers)
        with st.expander("Advanced"):
            a1, a2, a3 = st.columns(3)
            max_features = a1.select_slider("Max ORB keypoints", [1000, 2000, 5000, 8000, 12000], defaults.max_features)
            max_hamming = a2.slider("Max descriptor distance (bits)", 16, 96, defaults.max_hamming, 4)
            ransac = a3.slider("RANSAC threshold (px)", 1.0, 10.0, defaults.ransac_threshold_px, 0.5)
        run = st.form_submit_button("Run copy-move analysis", type="primary", icon=":material/play_arrow:")
    if run:
        params = CopyMoveParams(max_features=max_features, ratio=ratio, max_hamming=max_hamming,
                                min_distance_px=float(min_dist), min_inliers=min_inliers, ransac_threshold_px=ransac)
        with st.spinner("Detecting and matching keypoints..."):
            run_copy_move(case, params)

    if case.test_status.get("Copy-move analysis") == TEST_FAILED:
        callout("error", "Copy-move analysis failed", case.errors.get("Copy-move analysis", "Unknown error"))
    cm = case.copy_move
    if cm is None:
        callout("info", "Not run yet", "Press 'Run copy-move analysis' (defaults are a sensible starting point).")
        limitations(COPY_MOVE_LIMITATIONS)
        return

    kind = {"candidate_regions_found": "warn", "insufficient_features": "info"}.get(cm.status, "ok")
    title = {"candidate_regions_found": "Candidate duplicated regions (indicator, not proof)",
             "insufficient_features": "Inconclusive"}.get(cm.status, "No consistent duplication found")
    callout(kind, title, cm.status_text())
    m1, m2, m3 = st.columns(3)
    m1.metric("Keypoints", cm.keypoints_detected)
    m2.metric("Filtered matches", cm.candidate_matches)
    m3.metric("Consistent clusters", len(cm.clusters))
    if cm.visualization_rgb is not None:
        st.image(cm.visualization_rgb, use_container_width=True,
                 caption="Coloured: geometrically consistent clusters (outlines = matched areas). Grey: isolated matches.")
    if cm.clusters:
        st.markdown("#### Candidate region pairs")
        st.dataframe([
            {"#": i, "Matches": c.inliers, "Region A (x0,y0,x1,y1)": str(c.source_bbox),
             "Region B (x0,y0,x1,y1)": str(c.target_bbox), "Shift (px)": round(float(np.hypot(*c.translation)), 1),
             "Rotation (deg)": round(c.rotation_deg, 1), "Scale": round(c.scale, 2)}
            for i, c in enumerate(cm.clusters, start=1)
        ], hide_index=True, use_container_width=True)
    for note in cm.notes:
        st.caption(note)
    st.markdown("#### Parameters (for reproducibility)")
    kv_table(list(cm.params.as_display_dict().items()))
    limitations(COPY_MOVE_LIMITATIONS)
    next_button(PAGE_CONSISTENCY, "Next: Consistency Tools")


def page_consistency() -> None:
    hero("Exploratory views", "Image consistency tools",
         "Alternative renderings that can make inconsistencies easier to see: noise residual, local noise level, "
         "edges and colour channels. These are visual aids for a human - they produce no automatic verdict.")
    case = require_case()
    if case is None:
        return
    with st.form("consistency_form"):
        c1, c2, c3, c4 = st.columns(4)
        kernel = c1.selectbox("Median kernel", [3, 5, 7])
        amplification = c2.slider("Residual amplification", 1.0, 30.0, 8.0, 1.0)
        block = c3.select_slider("Noise-map block (px)", [8, 16, 32, 64], 16)
        sigma = c4.slider("Canny sigma", 0.1, 0.9, 0.33, 0.01)
        run = st.form_submit_button("Generate views", type="primary", icon=":material/play_arrow:")
    if run:
        with st.spinner("Rendering views..."):
            run_consistency(case, kernel, amplification, block, sigma)
    if case.test_status.get("Consistency tools") == TEST_FAILED:
        callout("error", "Failed", case.errors.get("Consistency tools", "Unknown error"))
    cons = case.consistency
    if cons is None:
        callout("info", "Not generated yet", "Press 'Generate views'. These tools are optional exploratory aids.")
        limitations(CONSISTENCY_LIMITATIONS)
        return
    t1, t2, t3, t4 = st.tabs(["Noise residual", "Noise-level map", "Edges", "Colour channels"])
    with t1:
        st.image(cons.noise_residual, caption="Indicator view: image minus median-filtered copy (amplified)",
                 use_container_width=True, clamp=True)
    with t2:
        st.image(cons.noise_level_map, caption="Indicator view: per-block residual standard deviation (bright = noisier)",
                 use_container_width=True)
    with t3:
        st.image(cons.edges, caption="Indicator view: Canny edges with automatic thresholds", use_container_width=True)
    with t4:
        names = list(cons.channels)
        for row in range(0, len(names), 3):
            cols = st.columns(3)
            for col, name in zip(cols, names[row:row + 3]):
                col.image(cons.channels[name], caption=f"{name} (indicator view)", use_container_width=True)
    kv_table(list(cons.parameters.items()))
    limitations(CONSISTENCY_LIMITATIONS)
    next_button(PAGE_RESULTS, "Next: Results & Report")


def page_results() -> None:
    hero("Results", "Analysis dashboard and forensic report",
         "A cautious, rule-based summary of every check. There is deliberately no 'real/fake' verdict and no "
         "confidence percentage: no validated, calibrated model backs these checks.")
    case = require_case()
    if case is None:
        return
    summary = build_summary(case.test_status, case.metadata, case.ela, case.copy_move, case.hash_comparison)
    kind = {POTENTIAL_INDICATORS: "warn", NO_INDICATORS: "ok", INCOMPLETE: "info"}[summary.status]
    callout(kind, summary.status, summary.explanation)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        card("File", case.filename, f"{case.validated.format} - {human_size(case.validated.size_bytes)}")
    with c2:
        card("SHA-256", case.sha256[:16] + "...",
             case.hash_comparison.status.replace("_", " ") if case.hash_comparison else "no reference supplied")
    with c3:
        card("Metadata", "EXIF present" if case.metadata and case.metadata.has_exif else "No EXIF",
             f"{len(case.metadata.indicators) if case.metadata else 0} indicator(s)")
    with c4:
        card("Indicators", str(len(summary.indicators)), "for human review")

    st.markdown("#### Check status")
    status_html = "".join(
        f'<tr><td>{esc(name)}</td><td>{pill(status, "ok" if status == TEST_COMPLETED else "error" if status == TEST_FAILED else "muted")}'
        f'{" " + esc(case.errors[name]) if name in case.errors else ""}</td></tr>'
        for name, status in case.test_status.items())
    st.markdown(f'<table class="is-kv">{status_html}</table>', unsafe_allow_html=True)
    if summary.incomplete:
        st.button("Run all remaining checks", on_click=goto, args=(PAGE_INTAKE,), icon=":material/play_arrow:",
                  help="Opens Evidence Intake, where 'Run all checks' runs every analysis with default settings.")

    col_a, col_b = st.columns(2, gap="large")
    with col_a:
        st.markdown("#### Indicators requiring review")
        if summary.indicators:
            for item in summary.indicators:
                callout("warn", item.source, item.detail)
        else:
            st.caption("None produced by the checks that ran.")
    with col_b:
        st.markdown("#### Neutral observations")
        for text in summary.observations or ["None recorded yet."]:
            callout("info", "Observation", text)

    st.markdown("#### Visual evidence")
    visuals = [(case.rgb, "Analysed image")]
    if case.ela is not None:
        visuals.append((case.ela.amplified_rgb, "ELA (amplified)"))
    if case.copy_move is not None and case.copy_move.visualization_rgb is not None:
        visuals.append((case.copy_move.visualization_rgb, "Copy-move matches"))
    cols = st.columns(len(visuals))
    for col, (image, caption) in zip(cols, visuals):
        col.image(image, caption=caption, use_container_width=True)

    st.divider()
    st.markdown("### Forensic PDF report")
    with st.form("report_form"):
        r1, r2 = st.columns(2)
        analyst = r1.text_input("Analyst name (optional)", max_chars=80)
        include_gps = r2.checkbox("Include GPS location in report", value=False,
                                  help="Off by default. Only enable if you are entitled to share the location.")
        include_views = r2.checkbox("Include consistency-tool images", value=True)
        notes = st.text_area("Analyst notes (optional, printed as user-entered text)", max_chars=2000)
        make_report = st.form_submit_button("Generate PDF report", type="primary", icon=":material/picture_as_pdf:")
    if make_report:
        if include_gps and case.metadata and case.metadata.has_gps:
            callout("warn", "GPS included", "The report will contain the image's location. Share it carefully.")
        try:
            with st.spinner("Building report in memory..."):
                st.session_state["report_pdf"] = generate_pdf_report(
                    case, ReportOptions(include_gps, analyst, notes, include_views), summary)
            log.info("Report generated for %s", case.investigation_id)
        except Exception as exc:
            invalidate_report()
            callout("error", "Report generation failed", f"{type(exc).__name__}: {exc}")
    pdf = st.session_state.get("report_pdf")
    if pdf:
        callout("ok", "Report ready", f"{human_size(len(pdf))}. Nothing is saved on the server; download it now.")
        st.download_button("Download PDF report", data=pdf, file_name=report_filename(case), mime="application/pdf",
                           type="primary", icon=":material/download:")
    elif summary.status == INCOMPLETE:
        st.caption("You can generate a report now; it will clearly list the checks that were not run.")


def page_about() -> None:
    hero(APP_NAME, APP_TAGLINE,
         "An educational, locally-run assistant for exploring forensic indicators in digital images. It supports a "
         "human examiner; it does not decide whether an image is real or fake.")
    st.markdown("#### Methods")
    kv_table([
        ("Validation", "Extension allow-list, magic bytes, Pillow verify + full decode, 20 MB / 50 MP limits"),
        ("Integrity", "SHA-256 over exact bytes (hashlib), constant-time reference comparison"),
        ("Metadata", "Pillow EXIF: IFD0, Exif IFD, GPS IFD; GPS redacted by default"),
        ("ELA", "In-memory JPEG recompression, amplified difference, tile ELA/texture outliers"),
        ("Copy-move", "ORB keypoints, k=3 self-matching, g2NN ratio test, RANSAC similarity clusters"),
        ("Consistency", "Median-filter noise residual, block noise map, Canny edges, RGB/YCrCb channels"),
        ("Report", "ReportLab PDF generated in memory; observations separated from conclusions"),
    ])
    st.markdown("#### Evidence handling")
    bullet_list(EVIDENCE_HANDLING_NOTES)
    st.markdown("#### Limitations")
    bullet_list(["Metadata can be stripped or forged; missing metadata is not proof of manipulation."]
                + ELA_LIMITATIONS + COPY_MOVE_LIMITATIONS + CONSISTENCY_LIMITATIONS)
    st.markdown("#### Privacy")
    bullet_list([
        "Runs on localhost; no image is uploaded to any third-party service or API; no API keys are used.",
        "Uploaded bytes live only in this browser session's memory and are discarded when you clear or close it.",
        "Logs record events (e.g. 'report generated') - never image contents or metadata values.",
        "Usage statistics collection by Streamlit is disabled in .streamlit/config.toml.",
    ])
    callout("warn", "Disclaimer", DISCLAIMER)


# --------------------------------------------------------------------------- main
def main() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    if "nav" not in st.session_state:
        st.session_state["nav"] = PAGE_INTAKE
    page = render_sidebar()
    {
        PAGE_INTAKE: page_intake,
        PAGE_METADATA: page_metadata,
        PAGE_INTEGRITY: page_integrity,
        PAGE_ELA: page_ela,
        PAGE_COPY_MOVE: page_copy_move,
        PAGE_CONSISTENCY: page_consistency,
        PAGE_RESULTS: page_results,
        PAGE_ABOUT: page_about,
    }[page]()
    st.caption(f"{APP_NAME} v{__version__} - educational tool. Indicators are not proof. Not certified for legal use.")


main()

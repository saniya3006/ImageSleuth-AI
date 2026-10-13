"""PDF forensic report generation with ReportLab.

The report is assembled entirely **in memory** (``io.BytesIO``): no temporary
files are written, so there is nothing to clean up and no copy of the evidence
is left on disk. The original upload is never embedded or modified - only
downscaled *visualisations* (JPEG-encoded illustrations) are included.

All text that originates from the user or from the file (filename, EXIF
values, notes) is XML-escaped before being placed in ReportLab paragraphs,
because ReportLab interprets a small markup language inside ``Paragraph``.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from datetime import datetime
from xml.sax.saxutils import escape

import numpy as np
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image as RLImage,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from . import APP_NAME, APP_TAGLINE, __version__
from .case import EvidenceCase
from .copy_move import COPY_MOVE_LIMITATIONS
from .ela import ELA_LIMITATIONS
from .integrity import verify_unchanged
from .metadata import gps_display
from .summary import AnalysisSummary, build_summary
from .utils import array_to_jpeg_bytes, format_timestamp, human_size, utc_now
from .visualizations import CONSISTENCY_LIMITATIONS

ACCENT = colors.HexColor("#0F766E")
ACCENT_LIGHT = colors.HexColor("#E6F4F1")
INK = colors.HexColor("#111827")
MUTED = colors.HexColor("#4B5563")
WARN_BG = colors.HexColor("#FFF7E6")
WARN_BORDER = colors.HexColor("#B45309")

DISCLAIMER = (
    "This report was produced by ImageSleuth AI, an educational image-forensics assistance tool. The "
    "techniques used (metadata inspection, SHA-256 hashing, Error Level Analysis, keypoint-based copy-move "
    "analysis and visual consistency aids) produce indicators that require expert human interpretation. They "
    "cannot conclusively determine whether an image is authentic or manipulated, no result has been "
    "calibrated against a validated forensic benchmark, and no probability or confidence score is implied. "
    "This software does not maintain a formal chain of custody and is not certified for legal, regulatory or "
    "evidentiary use; this report must not be presented as legally admissible forensic evidence."
)

EVIDENCE_HANDLING_NOTES = [
    "The SHA-256 digest was computed over the exact uploaded bytes before any parsing or analysis.",
    "All analyses operated on a separate decoded copy of the pixels; the original bytes were not modified.",
    "The uploaded file is held only in application memory for the session and is not stored permanently.",
    "No formal chain of custody (signed audit log, access control, trusted timestamping) is maintained.",
]


@dataclass
class ReportOptions:
    """User choices that affect report content."""

    include_gps: bool = False
    analyst_name: str = ""
    case_notes: str = ""
    include_consistency_images: bool = True


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("title", parent=base["Title"], textColor=INK, fontSize=20, leading=24, spaceAfter=2),
        "subtitle": ParagraphStyle("subtitle", parent=base["Normal"], textColor=MUTED, fontSize=9.5,
                                   alignment=TA_CENTER, spaceAfter=8),
        "h1": ParagraphStyle("h1", parent=base["Heading2"], textColor=ACCENT, fontSize=13.5, leading=17,
                             spaceBefore=10, spaceAfter=5),
        "h2": ParagraphStyle("h2", parent=base["Heading3"], textColor=INK, fontSize=11, leading=14,
                             spaceBefore=6, spaceAfter=3),
        "body": ParagraphStyle("body", parent=base["BodyText"], textColor=INK, fontSize=9.2, leading=12.5),
        "small": ParagraphStyle("small", parent=base["BodyText"], textColor=MUTED, fontSize=8, leading=10.5),
        "cell": ParagraphStyle("cell", parent=base["BodyText"], textColor=INK, fontSize=8.6, leading=11),
        "mono": ParagraphStyle("mono", parent=base["BodyText"], fontName="Courier", textColor=INK, fontSize=7.6,
                               leading=11),
        "bullet": ParagraphStyle("bullet", parent=base["BodyText"], textColor=INK, fontSize=9, leading=12,
                                 leftIndent=10, bulletIndent=0),
        "caption": ParagraphStyle("caption", parent=base["BodyText"], textColor=MUTED, fontSize=7.8,
                                  leading=10, alignment=TA_CENTER),
    }


def _p(text: str, style: ParagraphStyle) -> Paragraph:
    """Paragraph from *untrusted* plain text (escaped)."""
    return Paragraph(escape(str(text)), style)


def _kv_table(rows: list[tuple[str, str]], styles: dict[str, ParagraphStyle], mono_keys: set[str] | None = None) -> Table:
    mono_keys = mono_keys or set()
    data = [
        [_p(key, styles["cell"]), _p(value, styles["mono"] if key in mono_keys else styles["cell"])]
        for key, value in rows
    ]
    table = Table(data, colWidths=[52 * mm, 118 * mm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), ACCENT_LIGHT),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CBD5E1")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return table


def _bullets(items: list[str], styles: dict[str, ParagraphStyle]) -> list[Paragraph]:
    return [Paragraph(escape(item), styles["bullet"], bulletText="•") for item in items]


def _image(array: np.ndarray, width_mm: float, max_height_mm: float = 95) -> RLImage:
    # Illustrations only (the evidence itself is never embedded), so compact JPEG is appropriate.
    encoded = array_to_jpeg_bytes(array, max_dim=900, quality=85)
    height, width = array.shape[:2]
    draw_w = width_mm * mm
    draw_h = draw_w * height / width
    if draw_h > max_height_mm * mm:
        draw_h = max_height_mm * mm
        draw_w = draw_h * width / height
    return RLImage(io.BytesIO(encoded), width=draw_w, height=draw_h)


def _image_row(items: list[tuple[np.ndarray, str]], styles: dict[str, ParagraphStyle]) -> Table:
    col_width = 170 / len(items)
    cells = [[_image(array, col_width - 4, 70) for array, _ in items], [_p(cap, styles["caption"]) for _, cap in items]]
    table = Table(cells, colWidths=[col_width * mm] * len(items))
    table.setStyle(TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER"), ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
    return table


def _boxed(paragraphs: list[Paragraph], background: colors.Color, border: colors.Color) -> Table:
    table = Table([[paragraphs]], colWidths=[170 * mm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), background),
        ("BOX", (0, 0), (-1, -1), 0.8, border),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    return table


def _on_page(investigation_id: str):
    def draw(canvas, doc) -> None:  # noqa: ANN001 - ReportLab callback signature
        canvas.saveState()
        canvas.setStrokeColor(ACCENT)
        canvas.setLineWidth(1.2)
        canvas.line(20 * mm, A4[1] - 14 * mm, A4[0] - 20 * mm, A4[1] - 14 * mm)
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(MUTED)
        canvas.drawString(20 * mm, A4[1] - 12 * mm, f"{APP_NAME} - Educational forensic analysis report")
        canvas.drawRightString(A4[0] - 20 * mm, A4[1] - 12 * mm, investigation_id)
        canvas.drawString(20 * mm, 10 * mm, "Indicators only - not a determination of authenticity. Not legally certified.")
        canvas.drawRightString(A4[0] - 20 * mm, 10 * mm, f"Page {doc.page}")
        canvas.restoreState()
    return draw


def generate_pdf_report(
    case: EvidenceCase,
    options: ReportOptions | None = None,
    summary: AnalysisSummary | None = None,
    generated_at: datetime | None = None,
    compress: bool = True,
) -> bytes:
    """Build the PDF report for ``case`` and return it as bytes.

    Raises:
        ValueError: if the case holds no valid image.
    """
    if not case.is_valid or case.validated is None:
        raise ValueError("A report can only be generated for a successfully validated image.")
    options = options or ReportOptions()
    generated_at = generated_at or utc_now()
    summary = summary or build_summary(case.test_status, case.metadata, case.ela, case.copy_move, case.hash_comparison)
    styles = _styles()
    info = case.validated
    unchanged = verify_unchanged(case.original_bytes, case.sha256)

    story: list = [
        Paragraph(f"{APP_NAME} Forensic Analysis Report", styles["title"]),
        _p(APP_TAGLINE + " - educational use", styles["subtitle"]),
        _boxed([Paragraph("<b>Important:</b> " + escape(DISCLAIMER), styles["small"])], WARN_BG, WARN_BORDER),
        Spacer(1, 6),
    ]

    # 1. Case and file details
    story.append(Paragraph("1. Investigation and file details", styles["h1"]))
    rows = [
        ("Investigation ID", case.investigation_id),
        ("Report generated", format_timestamp(generated_at)),
        ("Evidence received", format_timestamp(case.received_at)),
        ("Analyst", options.analyst_name.strip() or "Not specified"),
        ("Original filename", case.filename),
        ("File size", f"{human_size(info.size_bytes)} ({info.size_bytes:,} bytes)"),
        ("Format (by content)", info.format),
        ("Dimensions", f"{info.width} x {info.height} px ({info.megapixels:.2f} MP)"),
        ("Colour mode", info.mode),
        ("Software version", f"{APP_NAME} {__version__}"),
    ]
    story.append(_kv_table(rows, styles))
    for warning in info.warnings:
        story.append(Spacer(1, 3))
        story.append(_p("Note: " + warning, styles["small"]))

    # 2. Overall summary
    story.append(Paragraph("2. Overall summary", styles["h1"]))
    story.append(_boxed([
        Paragraph(f"<b>{escape(summary.status)}</b>", styles["body"]),
        _p(summary.explanation, styles["small"]),
    ], ACCENT_LIGHT, ACCENT))
    story.append(Spacer(1, 4))
    story.append(_p(
        "Conclusion policy: this report records observations and indicators only. It draws NO conclusion "
        "about whether the image is authentic or manipulated.", styles["small"]))

    # 3. Integrity
    story.append(Paragraph("3. Integrity verification (SHA-256)", styles["h1"]))
    comparison = case.hash_comparison
    integrity_rows = [
        ("SHA-256 (at intake)", case.sha256),
        ("Re-verified at report time", "Yes - bytes unchanged" if unchanged else "NO - bytes changed during session"),
        ("Reference hash", comparison.reference if comparison and comparison.reference else "Not provided"),
        ("Comparison result", comparison.status.replace("_", " ").upper() if comparison else "NOT PROVIDED"),
    ]
    story.append(_kv_table(integrity_rows, styles, mono_keys={"SHA-256 (at intake)", "Reference hash"}))
    story.append(Spacer(1, 3))
    hash_note = (
        "A hash verifies byte-level equality only when a trusted reference exists; it does not by itself "
        "establish authenticity."
    )
    story.append(_p((comparison.message + " " + hash_note) if comparison else hash_note, styles["small"]))

    # 4. Metadata
    story.append(Paragraph("4. Metadata summary", styles["h1"]))
    metadata = case.metadata
    if metadata is None:
        story.append(_p("Metadata analysis was not performed.", styles["body"]))
    else:
        story.append(_kv_table(metadata.summary_rows(options.include_gps), styles))
        if metadata.error:
            story.append(_p("Parsing problem: " + metadata.error, styles["small"]))
        story.append(_p(
            "GPS data is " + ("INCLUDED at the user's explicit request." if options.include_gps
                              else "redacted by default for privacy."), styles["small"]))
        if metadata.gps and options.include_gps:
            story.append(_p("GPS: " + gps_display(metadata.gps, True), styles["small"]))

    # 5. Tests performed
    story.append(Paragraph("5. Tests performed", styles["h1"]))
    status_rows = [(name, status.upper() + (f" - {case.errors[name]}" if name in case.errors else ""))
                   for name, status in case.test_status.items()]
    story.append(_kv_table(status_rows, styles))

    # 6. ELA
    ela = case.ela
    story.append(Paragraph("6. Error Level Analysis", styles["h1"]))
    if ela is None:
        story.append(_p("Not run.", styles["body"]))
    else:
        story.append(_kv_table(list(ela.parameters().items()) + [
            ("Mean / max error", f"{ela.stats['mean_error']:.2f} / {ela.stats['max_error']:.0f}"),
            ("Outlier tiles", f"{len(ela.flagged_tiles)} of {ela.analysed_tiles} textured tiles"),
        ], styles))
        story.append(Spacer(1, 4))
        story.append(_image_row([(case.rgb, "Analysed image (stored orientation)"),
                                  (ela.amplified_rgb, f"ELA difference, amplified {ela.amplification:.1f}x"),
                                  (ela.outlier_overlay_rgb, "Outlier tiles (yellow), if any")], styles))
        story.extend(_bullets(["Warning: " + w for w in ela.warnings], styles))

    # 7. Copy-move
    cm = case.copy_move
    story.append(Paragraph("7. Copy-move analysis", styles["h1"]))
    if cm is None:
        story.append(_p("Not run.", styles["body"]))
    else:
        story.append(_kv_table(list(cm.params.as_display_dict().items()) + [
            ("Keypoints detected", str(cm.keypoints_detected)),
            ("Filtered candidate matches", str(cm.candidate_matches)),
            ("Consistent clusters", str(len(cm.clusters))),
        ], styles))
        story.append(Spacer(1, 3))
        story.append(_p("Result: " + cm.status_text(), styles["body"]))
        if cm.visualization_rgb is not None:
            story.append(Spacer(1, 4))
            story.append(KeepTogether([
                _image(cm.visualization_rgb, 150, 100),
                _p("Coloured lines/outlines: geometrically consistent match clusters. Grey: isolated matches.",
                   styles["caption"]),
            ]))
        story.extend(_bullets([f"Cluster #{i}: {c.describe()}" for i, c in enumerate(cm.clusters, 1)], styles))
        story.extend(_bullets(cm.notes, styles))

    # 8. Consistency tools
    cons = case.consistency
    if cons is not None:
        story.append(Paragraph("8. Exploratory consistency views", styles["h1"]))
        story.append(_kv_table(list(cons.parameters.items()), styles))
        if options.include_consistency_images:
            story.append(Spacer(1, 4))
            story.append(_image_row([(cons.noise_residual, "Noise residual"),
                                     (cons.noise_level_map, "Local noise-level map"),
                                     (cons.edges, "Canny edges")], styles))

    # 9. Observations vs indicators
    story.append(PageBreak())
    story.append(Paragraph("9. Findings", styles["h1"]))
    story.append(Paragraph("9.1 Indicators requiring human review", styles["h2"]))
    if summary.indicators:
        story.extend(_bullets([f"[{i.source}] {i.detail}" for i in summary.indicators], styles))
    else:
        story.append(_p("None produced by the checks that were run.", styles["body"]))
    story.append(Paragraph("9.2 Neutral observations", styles["h2"]))
    story.extend(_bullets(summary.observations or ["None recorded."], styles))
    story.append(Paragraph("9.3 Conclusions", styles["h2"]))
    story.append(_p("No conclusion about authenticity is drawn. Indicators and observations above are inputs for "
                    "a qualified examiner, not findings of fact.", styles["body"]))
    if options.case_notes.strip():
        story.append(Paragraph("9.4 Analyst notes (user-entered, not verified by the software)", styles["h2"]))
        story.append(_p(options.case_notes.strip(), styles["body"]))

    # 10. Limitations and evidence handling
    story.append(Paragraph("10. Limitations", styles["h1"]))
    story.append(Paragraph("Metadata", styles["h2"]))
    story.extend(_bullets([
        "Metadata can be removed, edited or forged with freely available tools. Missing metadata is not evidence "
        "of manipulation, and present metadata is not proof of authenticity."], styles))
    story.append(Paragraph("Error Level Analysis", styles["h2"]))
    story.extend(_bullets(ELA_LIMITATIONS, styles))
    story.append(Paragraph("Copy-move analysis", styles["h2"]))
    story.extend(_bullets(COPY_MOVE_LIMITATIONS, styles))
    story.append(Paragraph("Consistency views", styles["h2"]))
    story.extend(_bullets(CONSISTENCY_LIMITATIONS, styles))
    story.append(Paragraph("11. Evidence handling", styles["h1"]))
    story.extend(_bullets(EVIDENCE_HANDLING_NOTES, styles))
    story.append(Paragraph("12. Disclaimer", styles["h1"]))
    story.append(_boxed([_p(DISCLAIMER, styles["small"])], WARN_BG, WARN_BORDER))

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4, leftMargin=20 * mm, rightMargin=20 * mm, topMargin=20 * mm, bottomMargin=18 * mm,
        title=f"{APP_NAME} report {case.investigation_id}", author=APP_NAME, subject="Educational image forensics report",
        pageCompression=1 if compress else 0,
    )
    callback = _on_page(case.investigation_id)
    doc.build(story, onFirstPage=callback, onLaterPages=callback)
    return buffer.getvalue()


def report_filename(case: EvidenceCase) -> str:
    """Suggested download filename for a case report."""
    return f"ImageSleuth_Report_{case.investigation_id}.pdf"

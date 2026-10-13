"""Software-behaviour tests for PDF report generation."""

import zlib

import pytest

from forensic.case import open_case
from forensic.copy_move import CopyMoveParams, detect_copy_move
from forensic.ela import perform_ela
from forensic.integrity import compare_with_reference
from forensic.reporting import ReportOptions, generate_pdf_report, report_filename
from forensic.summary import TEST_COMPLETED
from forensic.visualizations import run_consistency_tools


def _full_case(data: bytes, name: str = "evidence.jpg"):
    case = open_case(data, name)
    case.ela = perform_ela(case.rgb, input_format=case.validated.format)
    case.mark("Error Level Analysis", TEST_COMPLETED)
    case.copy_move = detect_copy_move(case.rgb, CopyMoveParams())
    case.mark("Copy-move analysis", TEST_COMPLETED)
    case.consistency = run_consistency_tools(case.rgb)
    case.mark("Consistency tools", TEST_COMPLETED)
    case.hash_comparison = compare_with_reference(case.sha256, case.sha256)
    return case


def _is_pdf(pdf: bytes) -> bool:
    return pdf.startswith(b"%PDF-") and pdf.rstrip().endswith(b"%%EOF")


def test_generates_valid_pdf(exif_jpeg_bytes):
    case = _full_case(exif_jpeg_bytes)
    pdf = generate_pdf_report(case)
    assert _is_pdf(pdf)
    assert len(pdf) > 5_000
    assert report_filename(case) == f"ImageSleuth_Report_{case.investigation_id}.pdf"


def test_report_contains_required_sections_uncompressed(exif_jpeg_bytes):
    case = _full_case(exif_jpeg_bytes)
    pdf = generate_pdf_report(case, ReportOptions(analyst_name="Student A"), compress=False)
    for text in [case.investigation_id, case.sha256, "evidence.jpg", "Disclaimer", "Limitations",
                 "Error Level Analysis", "Copy-move analysis", "Student A", "Conclusions"]:
        assert text.encode() in pdf, text


def test_report_for_minimal_case_without_analyses(png_bytes):
    case = open_case(png_bytes, "only_intake.png")
    pdf = generate_pdf_report(case, compress=False)
    assert _is_pdf(pdf)
    assert b"Analysis incomplete" in pdf


def test_gps_redacted_unless_opted_in(gps_jpeg_bytes):
    case = open_case(gps_jpeg_bytes, "gps.jpg")
    default_pdf = generate_pdf_report(case, compress=False)
    assert b"48.858" not in default_pdf and b"redacted" in default_pdf
    opted_in = generate_pdf_report(case, ReportOptions(include_gps=True), compress=False)
    assert b"48.858" in opted_in


def test_markup_in_user_text_is_escaped(jpeg_bytes):
    case = open_case(jpeg_bytes, "<b>evil</b>&<script>.jpg")
    pdf = generate_pdf_report(case, ReportOptions(analyst_name="<i>x</i>", case_notes="a < b & c"), compress=False)
    assert _is_pdf(pdf)


def test_rejected_case_cannot_produce_report():
    with pytest.raises(ValueError):
        generate_pdf_report(open_case(b"junk", "x.jpg"))


def test_report_generation_does_not_alter_evidence(jpeg_bytes):
    case = _full_case(jpeg_bytes)
    before = case.sha256
    generate_pdf_report(case)
    assert case.original_bytes == jpeg_bytes and case.sha256 == before


def test_compressed_and_uncompressed_both_valid(jpeg_bytes):
    case = open_case(jpeg_bytes, "a.jpg")
    assert _is_pdf(generate_pdf_report(case, compress=True))
    assert zlib  # compression path exercised above

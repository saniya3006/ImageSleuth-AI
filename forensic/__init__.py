"""ImageSleuth AI forensic analysis package.

Every module in this package is independent of the Streamlit user interface so
that it can be unit-tested and reused from scripts. All analysis functions work
on in-memory copies of the decoded image; the original uploaded bytes are never
modified.

IMPORTANT: The outputs of this package are *indicators* intended to support a
human reviewer. None of the implemented techniques can, on their own, prove that
an image is authentic or manipulated.
"""

__all__ = [
    "case",
    "copy_move",
    "ela",
    "integrity",
    "metadata",
    "reporting",
    "summary",
    "utils",
    "validation",
    "visualizations",
]

__version__ = "1.0.0"
APP_NAME = "ImageSleuth AI"
APP_TAGLINE = "Intelligent Digital Image Forensics and Tampering Analysis System"

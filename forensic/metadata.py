"""EXIF / embedded metadata extraction using Pillow.

Metadata is easy to edit, strip or forge, so every observation produced here is
an *indicator for follow-up*, never proof:

* **Missing** EXIF is normal for screenshots, PNG exports, social-media
  downloads and privacy-scrubbed files.
* A **Software** tag naming an editor shows the file passed through that
  program (which may simply have resized or exported it).
* Differing timestamps can come from legitimate edits, camera clock settings or
  time-zone handling.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from PIL import ExifTags, Image

EXIF_IFD_POINTER = 0x8769
GPS_IFD_POINTER = 0x8825

UNAVAILABLE = "Unavailable"
MAX_VALUE_LENGTH = 120

ORIENTATION_LABELS = {
    1: "1 - Normal",
    2: "2 - Mirrored horizontally",
    3: "3 - Rotated 180°",
    4: "4 - Mirrored vertically",
    5: "5 - Mirrored horizontally, rotated 270° CW",
    6: "6 - Rotated 90° CW",
    7: "7 - Mirrored horizontally, rotated 90° CW",
    8: "8 - Rotated 270° CW",
}

# Programs whose name in the Software tag shows the file was processed/exported by an editor.
KNOWN_EDITING_SOFTWARE = (
    "photoshop", "lightroom", "gimp", "affinity", "paint.net", "pixelmator", "snapseed",
    "picsart", "canva", "photopea", "krita", "corel", "paintshop", "luminar", "capture one",
    "darktable", "rawtherapee", "facetune", "fotor", "photoscape", "inkscape", "imagemagick",
)

EXIF_DATETIME_FORMAT = "%Y:%m:%d %H:%M:%S"


@dataclass
class MetadataResult:
    """Structured metadata extracted from an image file."""

    has_exif: bool
    fields: dict[str, str] = field(default_factory=dict)
    camera_make: str = UNAVAILABLE
    camera_model: str = UNAVAILABLE
    datetime_original: str = UNAVAILABLE
    datetime_digitized: str = UNAVAILABLE
    datetime_modified: str = UNAVAILABLE
    software: str = UNAVAILABLE
    orientation: str = UNAVAILABLE
    gps: dict[str, Any] | None = None
    container_info: dict[str, str] = field(default_factory=dict)
    observations: list[str] = field(default_factory=list)
    indicators: list[str] = field(default_factory=list)
    error: str | None = None

    @property
    def has_gps(self) -> bool:
        return bool(self.gps)

    def summary_rows(self, include_gps: bool = False) -> list[tuple[str, str]]:
        """Key metadata as (label, value) rows; GPS is redacted unless ``include_gps``."""
        rows = [
            ("EXIF present", "Yes" if self.has_exif else "No"),
            ("Camera make", self.camera_make),
            ("Camera model", self.camera_model),
            ("Date/time original", self.datetime_original),
            ("Date/time digitized", self.datetime_digitized),
            ("Date/time modified (IFD0)", self.datetime_modified),
            ("Software", self.software),
            ("Orientation", self.orientation),
        ]
        rows.append(("GPS", gps_display(self.gps, include_gps)))
        return rows


def gps_display(gps: dict[str, Any] | None, include_gps: bool) -> str:
    """Human-readable GPS text, redacted unless the user explicitly opted in."""
    if not gps:
        return UNAVAILABLE
    if not include_gps:
        return "Present - redacted (enable 'include GPS' to show)"
    lat, lon = gps.get("latitude"), gps.get("longitude")
    if lat is None or lon is None:
        return "Present but incomplete (no usable coordinates)"
    text = f"{lat:.6f}, {lon:.6f}"
    if gps.get("altitude_m") is not None:
        text += f" (altitude {gps['altitude_m']:.1f} m)"
    return text


def _stringify(value: Any) -> str:
    """Convert an EXIF value to a short, printable string (binary blobs are summarised)."""
    if isinstance(value, bytes):
        stripped = value.rstrip(b"\x00")
        if stripped and all(32 <= b < 127 for b in stripped) and len(stripped) <= MAX_VALUE_LENGTH:
            return stripped.decode("ascii")
        return f"<binary data, {len(value)} bytes>"
    if isinstance(value, tuple):
        return ", ".join(_stringify(v) for v in value)
    text = str(value).replace("\x00", "").strip()
    if len(text) > MAX_VALUE_LENGTH:
        text = text[:MAX_VALUE_LENGTH] + "..."
    return text


def _rational_to_float(value: Any) -> float:
    """Convert Pillow IFDRational / (num, den) tuples / numbers to float."""
    if isinstance(value, tuple) and len(value) == 2:
        numerator, denominator = value
        return float(numerator) / float(denominator) if denominator else 0.0
    return float(value)


def dms_to_decimal(dms: Any, ref: str | None) -> float | None:
    """Convert EXIF degrees/minutes/seconds to signed decimal degrees."""
    try:
        degrees, minutes, seconds = (_rational_to_float(part) for part in dms)
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    decimal = degrees + minutes / 60.0 + seconds / 3600.0
    if ref and str(ref).strip().upper() in ("S", "W"):
        decimal = -decimal
    return round(decimal, 7)


def parse_gps(gps_ifd: dict[int, Any]) -> dict[str, Any] | None:
    """Decode a GPS IFD into decimal coordinates plus raw readable tags."""
    if not gps_ifd:
        return None
    named = {ExifTags.GPSTAGS.get(tag, f"GPSTag{tag}"): value for tag, value in gps_ifd.items()}
    result: dict[str, Any] = {"raw": {k: _stringify(v) for k, v in named.items()}}
    if "GPSLatitude" in named and "GPSLongitude" in named:
        result["latitude"] = dms_to_decimal(named["GPSLatitude"], named.get("GPSLatitudeRef"))
        result["longitude"] = dms_to_decimal(named["GPSLongitude"], named.get("GPSLongitudeRef"))
    if "GPSAltitude" in named:
        try:
            altitude = _rational_to_float(named["GPSAltitude"])
            if named.get("GPSAltitudeRef") in (1, b"\x01"):
                altitude = -altitude
            result["altitude_m"] = altitude
        except (TypeError, ValueError, ZeroDivisionError):
            pass
    return result


def _parse_exif_datetime(value: str) -> datetime | None:
    try:
        return datetime.strptime(value.strip(), EXIF_DATETIME_FORMAT)
    except (ValueError, AttributeError):
        return None


def extract_metadata(data: bytes) -> MetadataResult:
    """Extract EXIF and container metadata from image bytes.

    Never raises for malformed metadata: problems are reported in
    :attr:`MetadataResult.error` so the rest of the analysis can continue.
    """
    try:
        with Image.open(io.BytesIO(data)) as image:
            exif = image.getexif()
            container = {
                key: _stringify(value)
                for key, value in image.info.items()
                if key not in ("exif", "icc_profile", "xmp") and isinstance(value, (str, int, float, tuple, bytes))
            }
            if "icc_profile" in image.info:
                container["icc_profile"] = f"<embedded ICC colour profile, {len(image.info['icc_profile'])} bytes>"
            if "xmp" in image.info or "XML:com.adobe.xmp" in image.info:
                container["xmp"] = "<embedded XMP packet present>"
                container.pop("XML:com.adobe.xmp", None)
            base_tags = dict(exif.items())
            exif_ifd = dict(exif.get_ifd(EXIF_IFD_POINTER).items()) if EXIF_IFD_POINTER in exif else {}
            gps_ifd = dict(exif.get_ifd(GPS_IFD_POINTER).items()) if GPS_IFD_POINTER in exif else {}
    except Exception as exc:  # malformed EXIF blocks raise a variety of exceptions
        result = MetadataResult(has_exif=False, error=f"Metadata could not be parsed ({type(exc).__name__}).")
        result.observations.append(
            "Metadata parsing failed. This can happen with unusual or damaged metadata blocks and is "
            "not itself evidence of manipulation."
        )
        return result

    fields: dict[str, str] = {}
    for tag, value in {**base_tags, **exif_ifd}.items():
        if tag in (EXIF_IFD_POINTER, GPS_IFD_POINTER):
            continue
        name = ExifTags.TAGS.get(tag, f"Tag0x{tag:04X}")
        fields[name] = _stringify(value)

    result = MetadataResult(has_exif=bool(fields or gps_ifd), fields=fields, container_info=container)
    result.camera_make = fields.get("Make") or UNAVAILABLE
    result.camera_model = fields.get("Model") or UNAVAILABLE
    result.datetime_original = fields.get("DateTimeOriginal") or UNAVAILABLE
    result.datetime_digitized = fields.get("DateTimeDigitized") or UNAVAILABLE
    result.datetime_modified = fields.get("DateTime") or UNAVAILABLE
    result.software = fields.get("Software") or UNAVAILABLE
    raw_orientation = base_tags.get(0x0112)
    if raw_orientation is not None:
        result.orientation = ORIENTATION_LABELS.get(int(raw_orientation), str(raw_orientation))
    result.gps = parse_gps(gps_ifd)

    _add_observations(result)
    return result


def _add_observations(result: MetadataResult) -> None:
    """Populate neutral observations and (weak) indicators from extracted metadata."""
    if not result.has_exif:
        result.observations.append(
            "No EXIF metadata was found. This is common for screenshots, PNG exports, images downloaded "
            "from social media and privacy-scrubbed files. Missing metadata is NOT evidence of manipulation."
        )
        return

    if result.camera_make == UNAVAILABLE and result.camera_model == UNAVAILABLE:
        result.observations.append("EXIF is present but contains no camera make/model.")

    software = result.software
    if software != UNAVAILABLE:
        matched = [name for name in KNOWN_EDITING_SOFTWARE if name in software.lower()]
        if matched:
            result.indicators.append(
                f"The Software tag reads '{software}', indicating the file was processed or exported by "
                "image-editing software. Editors are also used for harmless adjustments (resize, crop, "
                "colour correction), so this shows processing, not deception."
            )
        else:
            result.observations.append(f"Software tag: '{software}' (often camera firmware or OS).")

    original = _parse_exif_datetime(result.datetime_original) if result.datetime_original != UNAVAILABLE else None
    modified = _parse_exif_datetime(result.datetime_modified) if result.datetime_modified != UNAVAILABLE else None
    if original and modified and modified != original:
        delta = modified - original
        result.indicators.append(
            f"The modification timestamp (DateTime) differs from DateTimeOriginal by {delta}. The file "
            "may have been re-saved or edited after capture; clock settings and some apps also cause this."
        )

    if result.has_gps:
        result.observations.append(
            "GPS location metadata is present. It is hidden in reports by default for privacy."
        )


def redact_gps_fields(fields: dict[str, str]) -> dict[str, str]:
    """Return a copy of an EXIF field dict with any GPS-related entries removed."""
    return {key: value for key, value in fields.items() if not key.upper().startswith("GPS")}

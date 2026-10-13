"""Evidence intake and per-investigation state.

Order of operations at intake (important for evidence handling):

1. The SHA-256 of the *exact uploaded bytes* is computed **first**, before any
   parsing or decoding.
2. The bytes are validated (type, size, structure).
3. A *separate decoded copy* of the pixels is created for analysis and marked
   read-only. The original ``bytes`` object is immutable in Python and is never
   re-encoded or modified.

This application does not implement a formal chain of custody (no signed
audit log, no access control, no trusted timestamping), so its results must
not be presented as legally admissible or certified forensic findings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from .copy_move import CopyMoveResult
from .ela import ELAResult
from .integrity import HashComparison, compute_sha256
from .metadata import MetadataResult, extract_metadata
from .summary import CORE_TESTS, TEST_COMPLETED, TEST_FAILED, TEST_NOT_RUN
from .utils import new_investigation_id, utc_now
from .validation import (
    DEFAULT_MAX_FILE_BYTES,
    ImageValidationError,
    ValidatedImage,
    decode_to_rgb,
    safe_filename,
    validate_image_bytes,
)
from .visualizations import ConsistencyResult

OPTIONAL_TESTS = ("Consistency tools",)


@dataclass
class EvidenceCase:
    """Everything known about one uploaded image during a session."""

    investigation_id: str
    received_at: datetime
    filename: str
    original_bytes: bytes
    sha256: str
    validated: ValidatedImage | None = None
    validation_error: str | None = None
    rgb: np.ndarray | None = None
    metadata: MetadataResult | None = None
    hash_comparison: HashComparison | None = None
    ela: ELAResult | None = None
    copy_move: CopyMoveResult | None = None
    consistency: ConsistencyResult | None = None
    test_status: dict[str, str] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        return self.validated is not None and self.rgb is not None

    def mark(self, test: str, status: str, error: str | None = None) -> None:
        self.test_status[test] = status
        if error:
            self.errors[test] = error
        else:
            self.errors.pop(test, None)


def open_case(data: bytes, filename: str, max_bytes: int = DEFAULT_MAX_FILE_BYTES) -> EvidenceCase:
    """Create an :class:`EvidenceCase` from uploaded bytes.

    Never raises for bad uploads: a rejected file yields a case with
    ``validation_error`` set so the UI can explain the problem.
    """
    data = bytes(data)  # defensive immutable copy
    case = EvidenceCase(
        investigation_id=new_investigation_id(),
        received_at=utc_now(),
        filename=safe_filename(filename),
        original_bytes=data,
        sha256=compute_sha256(data),  # step 1: hash before anything else touches the data
    )
    for test in (*CORE_TESTS, *OPTIONAL_TESTS):
        case.test_status[test] = TEST_NOT_RUN

    try:
        case.validated = validate_image_bytes(data, filename, max_bytes)
        rgb = decode_to_rgb(data)
        rgb.setflags(write=False)  # analysis code must work on copies
        case.rgb = rgb
    except ImageValidationError as exc:
        case.validation_error = str(exc)
        return case

    try:
        case.metadata = extract_metadata(data)
        case.mark("Metadata analysis", TEST_COMPLETED if case.metadata.error is None else TEST_FAILED,
                  case.metadata.error)
    except Exception as exc:  # pragma: no cover - extract_metadata already guards itself
        case.mark("Metadata analysis", TEST_FAILED, f"{type(exc).__name__}: {exc}")
    return case

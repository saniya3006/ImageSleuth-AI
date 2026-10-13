"""SHA-256 integrity verification.

A cryptographic hash answers exactly one question: *are these bytes identical
to the bytes that produced the reference hash?* It does **not** say whether the
picture is authentic. A doctored image has a perfectly valid SHA-256 of its own,
and an authentic photo re-saved by a messaging app gets a different hash even
though nobody edited its content. Hash comparison is only meaningful when the
reference value comes from a trusted source (for example, recorded at
acquisition time).
"""

from __future__ import annotations

import hashlib
import hmac
import string
from dataclasses import dataclass

SHA256_HEX_LENGTH = 64

STATUS_MATCH = "match"
STATUS_MISMATCH = "mismatch"
STATUS_INVALID = "invalid_reference"
STATUS_NOT_PROVIDED = "not_provided"


@dataclass(frozen=True)
class HashComparison:
    """Outcome of comparing a computed hash with a user-supplied reference."""

    status: str
    computed: str
    reference: str | None
    message: str

    @property
    def is_match(self) -> bool:
        return self.status == STATUS_MATCH


def compute_sha256(data: bytes) -> str:
    """Return the lowercase hexadecimal SHA-256 digest of ``data``.

    The digest is computed over the exact bytes supplied - no decoding,
    re-encoding or metadata stripping happens first.
    """
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise TypeError("compute_sha256 expects raw bytes")
    return hashlib.sha256(bytes(data)).hexdigest()


def normalize_hash(value: str) -> str:
    """Normalise user input: strip whitespace, an optional ``sha256:`` prefix, lowercase."""
    cleaned = "".join(value.split()).lower()
    if cleaned.startswith("sha256:"):
        cleaned = cleaned[len("sha256:"):]
    return cleaned


def is_valid_sha256(value: str) -> bool:
    """Return ``True`` when ``value`` is exactly 64 hexadecimal characters."""
    return len(value) == SHA256_HEX_LENGTH and all(c in string.hexdigits for c in value)


def compare_with_reference(computed: str, reference: str | None) -> HashComparison:
    """Compare a computed digest with an optional reference digest.

    Uses :func:`hmac.compare_digest` (constant-time comparison), which is good
    practice whenever secrets or integrity values are compared.
    """
    if reference is None or not reference.strip():
        return HashComparison(
            STATUS_NOT_PROVIDED,
            computed,
            None,
            "No reference hash supplied. The SHA-256 above identifies these exact bytes, "
            "but without a trusted reference there is nothing to verify it against.",
        )

    normalized = normalize_hash(reference)
    if not is_valid_sha256(normalized):
        return HashComparison(
            STATUS_INVALID,
            computed,
            normalized,
            "The reference value is not a valid SHA-256 digest (expected 64 hexadecimal characters).",
        )

    if hmac.compare_digest(computed.lower(), normalized):
        return HashComparison(
            STATUS_MATCH,
            computed,
            normalized,
            "MATCH: the uploaded bytes are identical to the bytes the reference hash was computed from. "
            "This confirms byte-level equality only - it is meaningful as far as the reference itself is "
            "trustworthy, and it does not establish that the image content is authentic.",
        )

    return HashComparison(
        STATUS_MISMATCH,
        computed,
        normalized,
        "MISMATCH: the uploaded bytes differ from the bytes the reference hash describes. Any change "
        "(re-saving, metadata edits, format conversion, transmission through a messaging app, or content "
        "editing) produces a different hash; a mismatch alone does not reveal what changed or why.",
    )


def verify_unchanged(data: bytes, expected_sha256: str) -> bool:
    """Re-hash ``data`` and confirm it still matches the hash recorded at intake."""
    return hmac.compare_digest(compute_sha256(data), expected_sha256.lower())

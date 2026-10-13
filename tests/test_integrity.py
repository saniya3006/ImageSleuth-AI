"""Software-behaviour tests for SHA-256 hashing and reference comparison."""

import hashlib

import pytest

from forensic.ela import perform_ela
from forensic.integrity import (
    STATUS_INVALID,
    STATUS_MATCH,
    STATUS_MISMATCH,
    STATUS_NOT_PROVIDED,
    compare_with_reference,
    compute_sha256,
    is_valid_sha256,
    normalize_hash,
    verify_unchanged,
)
from forensic.validation import decode_to_rgb

ABC_SHA256 = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"  # FIPS 180-2 test vector
EMPTY_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


def test_known_test_vectors():
    assert compute_sha256(b"abc") == ABC_SHA256
    assert compute_sha256(b"") == EMPTY_SHA256


def test_matches_hashlib_on_image_bytes(jpeg_bytes):
    assert compute_sha256(jpeg_bytes) == hashlib.sha256(jpeg_bytes).hexdigest()


def test_single_bit_change_changes_hash(jpeg_bytes):
    altered = bytearray(jpeg_bytes)
    altered[-10] ^= 0x01
    assert compute_sha256(bytes(altered)) != compute_sha256(jpeg_bytes)


def test_rejects_non_bytes():
    with pytest.raises(TypeError):
        compute_sha256("not bytes")  # type: ignore[arg-type]


def test_normalize_and_validate():
    assert normalize_hash("  SHA256:" + ABC_SHA256.upper() + "\n") == ABC_SHA256
    assert is_valid_sha256(ABC_SHA256)
    assert not is_valid_sha256(ABC_SHA256[:-1])
    assert not is_valid_sha256("z" * 64)


@pytest.mark.parametrize(
    "reference, expected",
    [
        (None, STATUS_NOT_PROVIDED),
        ("   ", STATUS_NOT_PROVIDED),
        ("1234", STATUS_INVALID),
        (ABC_SHA256, STATUS_MATCH),
        (ABC_SHA256.upper(), STATUS_MATCH),
        (EMPTY_SHA256, STATUS_MISMATCH),
    ],
)
def test_compare_with_reference(reference, expected):
    result = compare_with_reference(ABC_SHA256, reference)
    assert result.status == expected
    assert result.computed == ABC_SHA256
    assert result.message


def test_match_message_does_not_claim_authenticity():
    message = compare_with_reference(ABC_SHA256, ABC_SHA256).message.lower()
    assert "does not establish" in message


def test_analysis_does_not_modify_original_bytes(jpeg_bytes):
    """Evidence handling: running an analysis must leave the original bytes (and hash) unchanged."""
    before = compute_sha256(jpeg_bytes)
    perform_ela(decode_to_rgb(jpeg_bytes))
    assert verify_unchanged(jpeg_bytes, before)

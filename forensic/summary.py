"""Rule-based overall summary.

The summary deliberately has only three states and **no** score or
probability, because no validated, calibrated model backs this application:

* ``POTENTIAL_INDICATORS`` - at least one implemented check produced an
  indicator worth a human look.
* ``NO_INDICATORS`` - every core check completed and none produced an
  indicator. This does NOT mean the image is authentic.
* ``INCOMPLETE`` - a core check has not been run or failed, and nothing was
  flagged by the checks that did run.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .copy_move import CopyMoveResult
from .ela import ELAResult
from .integrity import STATUS_MISMATCH, HashComparison
from .metadata import MetadataResult

POTENTIAL_INDICATORS = "Potential indicators found"
NO_INDICATORS = "No indicators detected by implemented checks"
INCOMPLETE = "Analysis incomplete"

TEST_COMPLETED = "completed"
TEST_NOT_RUN = "not run"
TEST_FAILED = "failed"

CORE_TESTS = ("Metadata analysis", "Error Level Analysis", "Copy-move analysis")


@dataclass
class Indicator:
    """One observation that may warrant closer human examination."""

    source: str
    detail: str


@dataclass
class AnalysisSummary:
    status: str
    explanation: str
    indicators: list[Indicator] = field(default_factory=list)
    observations: list[str] = field(default_factory=list)
    incomplete: list[str] = field(default_factory=list)


def build_summary(
    test_status: dict[str, str],
    metadata: MetadataResult | None,
    ela: ELAResult | None,
    copy_move: CopyMoveResult | None,
    hash_comparison: HashComparison | None = None,
) -> AnalysisSummary:
    """Combine individual results into a cautious, three-state summary."""
    indicators: list[Indicator] = []
    observations: list[str] = []

    if hash_comparison is not None and hash_comparison.status == STATUS_MISMATCH:
        indicators.append(Indicator(
            "Integrity (SHA-256)",
            "The file bytes do not match the supplied reference hash, so this is not the exact file the "
            "reference describes. The difference could be anything from re-saving to editing.",
        ))
    elif hash_comparison is not None and hash_comparison.is_match:
        observations.append("SHA-256 matches the supplied reference hash (byte-identical to that reference).")

    if metadata is not None:
        indicators.extend(Indicator("Metadata", text) for text in metadata.indicators)
        observations.extend(metadata.observations)

    if ela is not None:
        if ela.has_localized_outliers:
            indicators.append(Indicator(
                "Error Level Analysis",
                f"{len(ela.flagged_tiles)} of {ela.analysed_tiles} textured tiles have an error-level-to-texture "
                f"ratio far above this image's typical level (robust z > {ela.z_threshold:g}). Such regions may "
                "have a different compression history; visual review is required.",
            ))
        elif ela.is_jpeg_input:
            observations.append("ELA tile statistics showed no localised outlier regions.")
        else:
            observations.append(
                f"ELA was run as a recompression experiment on a {ela.input_format} file; no ELA indicator is "
                "derived from non-JPEG inputs."
            )

    if copy_move is not None:
        if copy_move.found_candidates:
            for index, cluster in enumerate(copy_move.clusters, start=1):
                indicators.append(Indicator("Copy-move analysis", f"Candidate region pair #{index}: {cluster.describe()}."))
        else:
            observations.append(f"Copy-move: {copy_move.status_text()}")

    incomplete = [name for name in CORE_TESTS if test_status.get(name) != TEST_COMPLETED]
    if copy_move is not None and copy_move.status == "insufficient_features":
        incomplete.append("Copy-move analysis (insufficient features)")

    if indicators:
        status = POTENTIAL_INDICATORS
        explanation = (
            "One or more implemented checks produced indicators that deserve closer human examination. "
            "Indicators are not proof of manipulation; each has innocent explanations listed in the limitations."
        )
    elif incomplete:
        status = INCOMPLETE
        explanation = "Not all core checks have completed successfully: " + ", ".join(incomplete) + "."
    else:
        status = NO_INDICATORS
        explanation = (
            "The implemented checks completed without producing indicators. This does NOT establish that the "
            "image is authentic: skilled edits, smooth-region edits and edits followed by recompression can "
            "evade every method used here."
        )
    return AnalysisSummary(status, explanation, indicators, observations, incomplete)

"""Keypoint-based copy-move forgery analysis.

A *copy-move* forgery copies one part of an image and pastes it elsewhere in
the **same** image (e.g. to duplicate a crowd or hide an object). The pipeline:

1. **Detect** ORB keypoints (corner-like points) and compute 256-bit binary
   descriptors that describe the local appearance around each point.
2. **Self-match**: compare every descriptor against all others in the same
   image (brute force, Hamming distance), taking the 3 nearest neighbours.
3. **Filter** matches:
   * discard the trivial self-match;
   * *g2NN ratio test* - keep a match only if it is clearly better than the
     next-best candidate (``best < ratio x second``), which rejects
     ambiguous matches on repetitive texture;
   * absolute descriptor distance limit;
   * *minimum spatial distance* - matched points must be far apart, otherwise
     they are just neighbouring pixels on the same structure.
4. **Geometric consistency**: a genuinely copied region moves all its
   keypoints with the same transformation. RANSAC repeatedly fits a similarity
   transform (translation + rotation + uniform scale) and keeps only clusters
   with enough agreeing matches (*inliers*). Each accepted cluster is a
   *candidate* duplicated region pair.

Limitations are explained in :data:`COPY_MOVE_LIMITATIONS`. In short: flat
regions have no keypoints (copies of sky/walls are invisible to this method),
naturally repetitive scenes (windows, tiles, fences, text) can create
geometrically consistent matches without any forgery, and mirrored copies are
not modelled.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from .utils import resize_max_dim, to_gray

STATUS_CANDIDATES = "candidate_regions_found"
STATUS_NONE = "no_consistent_duplication"
STATUS_INSUFFICIENT = "insufficient_features"

CLUSTER_COLORS = [(255, 196, 0), (0, 220, 200), (255, 90, 160), (120, 200, 255), (180, 255, 120)]


@dataclass
class CopyMoveParams:
    """Tunable parameters (defaults chosen for clarity, not tuned on a benchmark)."""

    max_features: int = 5000
    ratio: float = 0.75
    max_hamming: int = 48
    min_distance_px: float = 40.0
    min_inliers: int = 8
    ransac_threshold_px: float = 4.0
    max_clusters: int = 5
    analysis_max_dim: int = 1600
    seed: int = 1234

    def validate(self) -> None:
        if not 100 <= self.max_features <= 50_000:
            raise ValueError("max_features must be between 100 and 50000")
        if not 0.3 <= self.ratio <= 0.95:
            raise ValueError("ratio must be between 0.3 and 0.95")
        if not 1 <= self.max_hamming <= 256:
            raise ValueError("max_hamming must be between 1 and 256")
        if self.min_distance_px < 5:
            raise ValueError("min_distance_px must be at least 5")
        if self.min_inliers < 4:
            raise ValueError("min_inliers must be at least 4")
        if self.ransac_threshold_px <= 0:
            raise ValueError("ransac_threshold_px must be positive")
        if not 1 <= self.max_clusters <= 20:
            raise ValueError("max_clusters must be between 1 and 20")
        if self.analysis_max_dim < 256:
            raise ValueError("analysis_max_dim must be at least 256")

    def as_display_dict(self) -> dict[str, str]:
        return {
            "Feature detector": f"ORB, up to {self.max_features} keypoints",
            "Matcher": "Brute-force Hamming, k=3 self-matching",
            "Ratio test (g2NN)": f"best < {self.ratio:.2f} x second-best",
            "Max descriptor distance": f"{self.max_hamming} bits",
            "Min spatial distance": f"{self.min_distance_px:.0f} px",
            "Geometric model": "Similarity transform (RANSAC)",
            "RANSAC reprojection threshold": f"{self.ransac_threshold_px:.1f} px",
            "Min inliers per cluster": str(self.min_inliers),
            "Analysis resolution": f"longest side <= {self.analysis_max_dim} px",
            "RNG seed": str(self.seed),
        }


@dataclass
class MatchCluster:
    """A group of matches that agree on one geometric transformation."""

    source_points: np.ndarray  # (N, 2) in original-image pixel coordinates
    target_points: np.ndarray
    translation: tuple[float, float]
    rotation_deg: float
    scale: float

    @property
    def inliers(self) -> int:
        return int(len(self.source_points))

    @staticmethod
    def _bbox(points: np.ndarray) -> tuple[int, int, int, int]:
        x0, y0 = points.min(axis=0)
        x1, y1 = points.max(axis=0)
        return int(x0), int(y0), int(x1), int(y1)

    @property
    def source_bbox(self) -> tuple[int, int, int, int]:
        return self._bbox(self.source_points)

    @property
    def target_bbox(self) -> tuple[int, int, int, int]:
        return self._bbox(self.target_points)

    def describe(self) -> str:
        dx, dy = self.translation
        return (
            f"{self.inliers} consistent matches; region near {self.source_bbox[:2]} maps to near "
            f"{self.target_bbox[:2]} (shift {math.hypot(dx, dy):.0f} px, rotation {self.rotation_deg:.1f} deg, "
            f"scale {self.scale:.2f})"
        )


@dataclass
class CopyMoveResult:
    """Output of :func:`detect_copy_move`."""

    status: str
    params: CopyMoveParams
    keypoints_detected: int
    candidate_matches: int
    clusters: list[MatchCluster] = field(default_factory=list)
    visualization_rgb: np.ndarray | None = None
    analysis_scale: float = 1.0
    notes: list[str] = field(default_factory=list)

    @property
    def found_candidates(self) -> bool:
        return self.status == STATUS_CANDIDATES

    def status_text(self) -> str:
        if self.status == STATUS_CANDIDATES:
            return (
                f"{len(self.clusters)} candidate duplicated region pair(s) with geometrically consistent "
                "matches. Inspect visually: repetitive real-world structures can also produce this."
            )
        if self.status == STATUS_INSUFFICIENT:
            return (
                "Too few distinctive keypoints for a meaningful search. Copy-move cannot be assessed by this "
                "method (smooth or very small images); this is NOT a clean result."
            )
        return (
            "No geometrically consistent duplicated regions were found by this method. Copies of smooth or "
            "low-texture areas, heavily transformed copies and mirrored copies may still go undetected."
        )


def _orb_features(gray: np.ndarray, max_features: int) -> tuple[list[cv2.KeyPoint], np.ndarray | None]:
    orb = cv2.ORB_create(nfeatures=max_features, scaleFactor=1.2, nlevels=8, edgeThreshold=15, patchSize=31,
                         fastThreshold=10)
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    return list(keypoints), descriptors


def _candidate_pairs(
    keypoints: list[cv2.KeyPoint], descriptors: np.ndarray, params: CopyMoveParams
) -> list[tuple[int, int]]:
    """Self-match descriptors and apply ratio, Hamming and spatial-distance filters."""
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn = matcher.knnMatch(descriptors, descriptors, k=3)
    points = np.array([kp.pt for kp in keypoints], dtype=np.float32)

    pairs: set[tuple[int, int]] = set()
    for neighbours in knn:
        others = [m for m in neighbours if m.trainIdx != m.queryIdx]
        if len(others) < 2:
            continue
        best, second = others[0], others[1]
        if best.distance > params.max_hamming:
            continue
        if best.distance >= params.ratio * second.distance:
            continue
        i, j = best.queryIdx, best.trainIdx
        if float(np.linalg.norm(points[i] - points[j])) < params.min_distance_px:
            continue
        pairs.add((min(i, j), max(i, j)))  # (i, j) and (j, i) are the same pair
    return sorted(pairs)


def _cluster_pairs(
    points: np.ndarray, pairs: list[tuple[int, int]], params: CopyMoveParams
) -> list[MatchCluster]:
    """Iteratively extract groups of pairs consistent with one similarity transform.

    Each pair is inserted in both directions (A->B and B->A) because we do not
    know which side is the original; once a cluster is accepted, all its pairs
    are removed in both directions before searching for the next cluster.
    """
    if len(pairs) < params.min_inliers:
        return []
    pair_arr = np.array(pairs, dtype=np.int64)
    src = np.vstack([points[pair_arr[:, 0]], points[pair_arr[:, 1]]]).astype(np.float32)
    dst = np.vstack([points[pair_arr[:, 1]], points[pair_arr[:, 0]]]).astype(np.float32)
    pair_ids = np.concatenate([np.arange(len(pairs)), np.arange(len(pairs))])
    active = np.ones(len(src), dtype=bool)

    clusters: list[MatchCluster] = []
    attempts = 0
    while len(clusters) < params.max_clusters and active.sum() >= params.min_inliers and attempts < params.max_clusters * 3:
        attempts += 1
        idx = np.flatnonzero(active)
        matrix, mask = cv2.estimateAffinePartial2D(
            src[idx], dst[idx], method=cv2.RANSAC, ransacReprojThreshold=params.ransac_threshold_px,
            maxIters=3000, confidence=0.995, refineIters=10,
        )
        if matrix is None or mask is None:
            break
        inlier_idx = idx[mask.ravel().astype(bool)]
        inlier_pairs = np.unique(pair_ids[inlier_idx])
        if len(inlier_pairs) < params.min_inliers:
            break

        a, b = float(matrix[0, 0]), float(matrix[1, 0])
        scale = math.hypot(a, b)
        rotation = math.degrees(math.atan2(b, a))
        tx, ty = float(matrix[0, 2]), float(matrix[1, 2])
        # Remove these pairs (both directions) regardless of acceptance, to make progress.
        active &= ~np.isin(pair_ids, inlier_pairs)
        if not 0.5 <= scale <= 2.0:
            continue  # implausible geometry: treat as coincidental agreement

        # Keep one direction per pair (the one that fits the transform) for display.
        chosen: dict[int, int] = {}
        for k in inlier_idx:
            chosen.setdefault(int(pair_ids[k]), int(k))
        rows = np.array(list(chosen.values()))
        source_pts, target_pts = src[rows], dst[rows]
        if float(np.median(np.linalg.norm(target_pts - source_pts, axis=1))) < params.min_distance_px:
            continue
        clusters.append(MatchCluster(source_pts, target_pts, (tx, ty), rotation, scale))
    return clusters


def _draw(rgb: np.ndarray, clusters: list[MatchCluster], unclustered: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    canvas = (rgb.astype(np.float32) * 0.6).astype(np.uint8)
    thickness = max(1, round(max(rgb.shape[:2]) / 800))
    for p, q in unclustered:
        cv2.line(canvas, tuple(int(v) for v in p), tuple(int(v) for v in q), (110, 110, 110), 1, cv2.LINE_AA)
    for index, cluster in enumerate(clusters):
        color = CLUSTER_COLORS[index % len(CLUSTER_COLORS)]
        for p, q in zip(cluster.source_points, cluster.target_points):
            cv2.line(canvas, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, thickness, cv2.LINE_AA)
            cv2.circle(canvas, (int(p[0]), int(p[1])), 2 + thickness, color, -1, cv2.LINE_AA)
            cv2.circle(canvas, (int(q[0]), int(q[1])), 2 + thickness, color, -1, cv2.LINE_AA)
        for pts in (cluster.source_points, cluster.target_points):
            if len(pts) >= 3:
                hull = cv2.convexHull(pts.astype(np.int32))
                cv2.polylines(canvas, [hull], True, color, thickness + 1, cv2.LINE_AA)
        x0, y0, _, _ = cluster.source_bbox
        cv2.putText(canvas, f"#{index + 1}", (x0, max(14, y0 - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5 + thickness * 0.2, color, thickness, cv2.LINE_AA)
    return canvas


def detect_copy_move(rgb: np.ndarray, params: CopyMoveParams | None = None) -> CopyMoveResult:
    """Search an RGB ``uint8`` image for candidate copy-moved regions.

    The input array is not modified. Large images are downscaled for speed and
    all coordinates are mapped back to the original resolution.
    """
    params = params or CopyMoveParams()
    params.validate()
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("detect_copy_move expects an RGB uint8 array of shape (H, W, 3)")

    cv2.setRNGSeed(params.seed)  # makes RANSAC reproducible
    working, scale = resize_max_dim(rgb, params.analysis_max_dim)
    gray = to_gray(working)
    keypoints, descriptors = _orb_features(gray, params.max_features)

    result = CopyMoveResult(
        status=STATUS_INSUFFICIENT, params=params, keypoints_detected=len(keypoints), candidate_matches=0,
        analysis_scale=scale,
    )
    if scale < 1.0:
        result.notes.append(
            f"Image was downscaled by a factor of {scale:.3f} for feature analysis; coordinates are reported "
            "in original-resolution pixels."
        )
    min_keypoints = max(2 * params.min_inliers, 30)
    if descriptors is None or len(keypoints) < min_keypoints:
        result.visualization_rgb = _draw(rgb, [], [])
        result.notes.append(f"Only {len(keypoints)} keypoints detected (at least {min_keypoints} needed).")
        return result

    pairs = _candidate_pairs(keypoints, descriptors, params)
    result.candidate_matches = len(pairs)
    points = np.array([kp.pt for kp in keypoints], dtype=np.float32) / scale  # back to original coordinates
    # Spatial distance was checked at analysis scale; also apply the threshold at analysis scale for RANSAC.
    scaled_params = CopyMoveParams(**{**params.__dict__, "min_distance_px": params.min_distance_px / scale,
                                      "ransac_threshold_px": params.ransac_threshold_px / scale})
    clusters = _cluster_pairs(points, pairs, scaled_params)
    result.clusters = clusters
    result.status = STATUS_CANDIDATES if clusters else STATUS_NONE

    clustered = {tuple(np.round(p, 1)) for c in clusters for p in c.source_points} | {
        tuple(np.round(p, 1)) for c in clusters for p in c.target_points
    }
    unclustered = [
        (points[i], points[j]) for i, j in pairs
        if tuple(np.round(points[i], 1)) not in clustered and tuple(np.round(points[j], 1)) not in clustered
    ][:400]
    result.visualization_rgb = _draw(rgb, clusters, unclustered)
    if pairs and not clusters:
        result.notes.append(
            f"{len(pairs)} individual descriptor matches survived filtering but none formed a geometrically "
            f"consistent group of at least {params.min_inliers}; isolated matches are expected in most images "
            "and are shown in grey."
        )
    return result


COPY_MOVE_LIMITATIONS = [
    "Only keypoint-rich (textured, cornered) areas can be matched. Copies of smooth regions such as sky, "
    "walls or skin usually produce no keypoints and cannot be detected by this method.",
    "Naturally repetitive content - windows, tiles, fences, text, patterned fabric, logos - can produce "
    "geometrically consistent matches without any manipulation.",
    "Mirrored (flipped) copies, strong scaling, heavy blur/noise or aggressive recompression of the "
    "pasted region reduce the number of matches and can hide a forgery.",
    "A region copied from a DIFFERENT image (splicing) is outside the scope of copy-move analysis.",
    "Parameters are reasonable defaults, not values calibrated on a forensic benchmark; no probability or "
    "confidence score is produced.",
]

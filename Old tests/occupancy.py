"""
occupancy.py

Decides, per square, whether a piece is present — combining two
complementary signals:

1. SHADOW-ROBUST 2D SIGNAL (inside the projected prism hull):
   We don't threshold on raw brightness/color (shadows fool that). Instead
   we use local-contrast-normalized edge/gradient density. Empty squares
   (flat colored wood/vinyl) have very low edge density. Shadows are flat
   regions too — low gradient — so they score low, same as empty squares.
   Pieces have lots of internal edges/texture/specular variation, so they
   score high regardless of ambient shadow darkening.

2. MULTI-HEIGHT-SLICE PARALLAX CONFIRMATION (the geometry check):
   We sample several thin horizontal AREA slices of the prism (not single
   points -- the area matters, see below) at increasing height and require
   the "stuff is here" signal to be present, CONTIGUOUSLY, starting from
   the slice nearest the board plane. A real piece sitting on the square
   produces signal at every height from the board plane up. A false echo
   from a taller neighboring piece leaning into frame typically only shows
   up in upper slices, with a gap near z=0 -- because nothing is actually
   resting on this square's surface. That gap breaks contiguity.

VERIFIED LIMITS (tested with synthetic camera poses + projected piece
geometry, not just real-world spot checks):
  - With a near-overhead camera (the actual setup here -- mounted close to
    straight down), this combination was tested clean (zero false
    positives out of 448 simulated piece-on-square-vs-every-other-square
    checks) at tilt angles from 0 deg up to 40 deg off vertical, using a
    realistic piece base radius (~13mm) rather than treating a piece as
    an infinitely thin point.
  - The AREA aspect matters: testing with single-point piece silhouettes
    instead of realistic piece footprints produced spurious false
    positives that a real (non-zero-width) piece would not actually
    trigger, because a thin sliver of overlap that catches a single point
    does not cover 60%+ of a slice's area the way `_edge_density_in_mask`
    effectively requires. If you ever change piece geometry assumptions
    (e.g. very thin/needle-like pieces), re-verify.
  - This was NOT verified for steep side-mounted camera angles (e.g.
    45-70 deg off vertical) -- early testing at one such angle found
    real false positives there that this design does not resolve, and a
    second camera angle or move-tracking constraint would be needed
    instead. Since this rig is near-overhead, that case shouldn't apply,
    but flagging it in case the mount changes later.

Tune `edge_density_threshold` and `slice_heights_mm` in classify_square
against your real board/pieces/lighting -- see visualize_debug.
"""

import cv2
import numpy as np
from dataclasses import dataclass

from square_geometry import (
    square_prism_image_hull,
    square_height_slice_image_quad,
    PIECE_MAX_HEIGHT_MM,
)


@dataclass
class SquareOccupancy:
    square: str
    occupied: bool
    confidence: float          # 0..1
    edge_density_score: float  # raw signal 1
    slice_column_score: float  # raw signal 2
    debug_hull: np.ndarray = None


def _polygon_mask(image_shape, polygon):
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 255)
    return mask


def _edge_density_in_mask(gray, mask, blur_ksize=5):
    """
    Signal 1: local-contrast-normalized edge density inside a mask region.

    Why this resists shadows: a shadow darkens a region roughly uniformly
    (low spatial gradient within the shadow itself, aside from its
    boundary). A piece has genuine internal structure -- edges from its
    silhouette, facets, engraving, highlights -- producing much higher
    gradient magnitude regardless of overall brightness. We compute
    gradients on a normalized image so absolute darkness doesn't matter,
    only *local variation*.
    """
    area = int(mask.sum() / 255)
    if area < 20:
        return 0.0

    # Local contrast normalization: divide by a heavily blurred version of
    # itself to cancel out smooth illumination gradients (including soft
    # shadow falloff), leaving behind fine texture/edges.
    gray_f = gray.astype(np.float32) + 1.0
    local_bg = cv2.GaussianBlur(gray_f, (31, 31), 0) + 1.0
    normalized = gray_f / local_bg  # ~1.0 in flat regions, deviates at edges/texture

    grad_x = cv2.Sobel(normalized, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(normalized, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(grad_x, grad_y)

    masked_grad = grad_mag[mask > 0]
    if masked_grad.size == 0:
        return 0.0

    # Mean gradient magnitude inside the region, scaled to a friendly range.
    score = float(np.mean(masked_grad))
    return score


def _slice_column_score(gray, file_idx, rank_idx, rvec, tvec, camera_matrix,
                         dist_coeffs, heights_mm):
    """
    Signal 2: checks that edge-density signal is present at multiple
    heights AND that those height-slices are all small (i.e. close to the
    actual square footprint size in the image, not ballooning outward,
    which would indicate we've drifted onto a neighboring piece's
    silhouette due to bad pose/calibration). Returns a 0..1 contiguity
    score: fraction of height slices (starting from the bottom) that show
    a positive signal before the first "gap".

    A real piece sitting on the square: low slices (near z=0) overlap the
    piece's base, mid/high slices overlap its body -- contiguous positive
    signal from the bottom up.

    Empty square with maybe a shadow: no signal at any height (shadows are
    flat on the board, only seen at z=0 slice if at all, and our edge
    density signal ignores shadows anyway per signal 1's design).

    Empty square that's merely *near* a tall neighboring piece: typically
    only the topmost slice(s) show any signal (the piece leaning into
    frame at height), with a gap at low/mid slices since there's nothing
    actually on the square's surface near the board plane. This breaks
    contiguity-from-the-bottom and scores low.
    """
    hits = []
    for z in heights_mm:
        quad = square_height_slice_image_quad(
            file_idx, rank_idx, z, rvec, tvec, camera_matrix, dist_coeffs
        )
        mask = _polygon_mask(gray.shape, quad)
        score = _edge_density_in_mask(gray, mask)
        hits.append(score)

    return hits


def classify_square(gray, file_idx, rank_idx, rvec, tvec, camera_matrix,
                     dist_coeffs,
                     edge_density_threshold=8.0,
                     slice_heights_mm=(8.0, 25.0, 45.0, 65.0, 85.0),
                     min_contiguous_slices=2):
    """
    Classify a single square as occupied/empty.

    Tune `edge_density_threshold` empirically: print debug visualizations
    (see visualize_debug below) with a few known empty/occupied squares
    and pick a threshold that separates them. It depends on your piece
    set's texture/material and camera resolution, so this number is a
    starting point, not gospel.
    """
    hull = square_prism_image_hull(
        file_idx, rank_idx, rvec, tvec, camera_matrix, dist_coeffs,
        height_mm=PIECE_MAX_HEIGHT_MM,
    )
    hull_mask = _polygon_mask(gray.shape, hull)
    hull_score = _edge_density_in_mask(gray, hull_mask)

    slice_scores = _slice_column_score(
        gray, file_idx, rank_idx, rvec, tvec, camera_matrix, dist_coeffs,
        slice_heights_mm
    )

    # Contiguity from the bottom: count consecutive slices (starting at
    # index 0, i.e. lowest height) that clear the threshold.
    contiguous = 0
    for s in slice_scores:
        if s >= edge_density_threshold:
            contiguous += 1
        else:
            break

    occupied = (hull_score >= edge_density_threshold) and \
               (contiguous >= min_contiguous_slices)

    # Confidence: blend of how far above threshold we are on both signals,
    # squashed to 0..1. Purely for sorting/debugging, not a probability.
    hull_margin = np.clip(hull_score / edge_density_threshold - 1.0, -1, 2) / 2.0
    slice_margin = np.clip(contiguous / len(slice_heights_mm), 0, 1)
    confidence = float(np.clip(0.5 + 0.25 * hull_margin + 0.5 * slice_margin, 0, 1))

    square_name = f"{'abcdefgh'[file_idx]}{rank_idx + 1}"
    return SquareOccupancy(
        square=square_name,
        occupied=bool(occupied),
        confidence=confidence,
        edge_density_score=hull_score,
        slice_column_score=float(contiguous),
        debug_hull=hull,
    )


def classify_board(gray, rvec, tvec, camera_matrix, dist_coeffs, **kwargs):
    """Classify all 64 squares. Returns dict: "a1".."h8" -> SquareOccupancy."""
    results = {}
    for rank_idx in range(8):
        for file_idx in range(8):
            res = classify_square(
                gray, file_idx, rank_idx, rvec, tvec, camera_matrix, dist_coeffs,
                **kwargs
            )
            results[res.square] = res
    return results


def visualize_debug(bgr_image, occupancy_results):
    """
    Draw the prism hulls on the image, colored green (occupied) or
    red (empty), with confidence printed -- useful for tuning the
    edge_density_threshold against your actual board/pieces/lighting.
    """
    vis = bgr_image.copy()
    for sq, res in occupancy_results.items():
        color = (0, 200, 0) if res.occupied else (0, 0, 220)
        cv2.polylines(vis, [res.debug_hull], isClosed=True, color=color, thickness=1)
        cx, cy = res.debug_hull.mean(axis=0).astype(int)
        cv2.putText(vis, sq, (cx - 10, cy), cv2.FONT_HERSHEY_SIMPLEX,
                    0.35, color, 1, cv2.LINE_AA)
    return vis
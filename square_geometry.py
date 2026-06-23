"""
square_geometry.py

This is the module that actually solves your parallax problem.

Instead of treating each square as a flat 2D quad in the image (which is
what gets fooled when a tall piece on a nearer square leans into a farther
square's projected region), we define each square's "occupancy volume" as
a 3D prism: the square's footprint on the board, extruded upward to roughly
the height of your tallest piece (e.g. king/queen).

We then project THIS 3D volume into the image using the camera pose from
board_pose.py. The resulting image-space polygon is the region we sample
for the coarse "is anything tall-ish near this square" signal.

IMPORTANT, found via testing: at steep camera angles, adjacent squares'
projected prism hulls DO overlap in image space, and a piece on square N
can echo into square N+1's silhouette. The hull alone is necessary-but-
NOT-sufficient. For a near-overhead camera (verified clean 0-40 degrees
off vertical with realistic piece footprints) this is resolved by
occupancy.py's height-slice contiguity check. For steep side-mounted
angles it is NOT reliably resolved by this method -- see occupancy.py's
module docstring for the verified scope.
"""

import cv2
import numpy as np

SQUARE_SIZE_MM = 35.0
PIECE_MAX_HEIGHT_MM = 95.0   # set to your tallest piece (typically the king)
                              # err on the high side; we sample multiple
                              # height slices anyway (see occupancy.py)

FILES = "abcdefgh"
RANKS = "12345678"


def square_origin_mm(file_idx, rank_idx):
    """
    Bottom-left corner (z=0) of a square in board coordinates, given
    0-indexed file (0=a) and rank (0=1). Matches the board frame defined
    in board_pose.py: origin at a1's outer corner.
    """
    return np.array([file_idx * SQUARE_SIZE_MM, rank_idx * SQUARE_SIZE_MM, 0.0])


def square_footprint_3d(file_idx, rank_idx, shrink_mm=3.0):
    """
    The 4 ground-plane (z=0) corners of a square, optionally shrunk inward
    slightly (shrink_mm) so the sampling region doesn't bleed into
    neighboring squares/gridlines — helpful since board printing/cutting
    is never perfectly precise.
    """
    x0 = file_idx * SQUARE_SIZE_MM + shrink_mm
    x1 = (file_idx + 1) * SQUARE_SIZE_MM - shrink_mm
    y0 = rank_idx * SQUARE_SIZE_MM + shrink_mm
    y1 = (rank_idx + 1) * SQUARE_SIZE_MM - shrink_mm
    return np.array([
        [x0, y0, 0.0],
        [x1, y0, 0.0],
        [x1, y1, 0.0],
        [x0, y1, 0.0],
    ], dtype=np.float64)


def square_prism_3d(file_idx, rank_idx, height_mm=PIECE_MAX_HEIGHT_MM, shrink_mm=3.0):
    """
    Full 3D prism (8 corners: 4 on the board plane, 4 at height_mm above)
    for a square. This is the volume we project to get the occupancy
    sampling region.
    """
    base = square_footprint_3d(file_idx, rank_idx, shrink_mm)
    top = base.copy()
    top[:, 2] = height_mm
    return np.vstack([base, top])  # shape (8, 3)


def project_points(points_3d, rvec, tvec, camera_matrix, dist_coeffs):
    """Project Nx3 board-frame points into Nx2 image pixel coordinates."""
    pts = np.asarray(points_3d, dtype=np.float64).reshape(-1, 1, 3)
    projected, _ = cv2.projectPoints(pts, rvec, tvec, camera_matrix, dist_coeffs)
    return projected.reshape(-1, 2)


def square_prism_image_hull(file_idx, rank_idx, rvec, tvec, camera_matrix,
                             dist_coeffs, height_mm=PIECE_MAX_HEIGHT_MM,
                             shrink_mm=3.0):
    """
    Project a square's 3D prism into the image and return the convex hull
    of its 8 projected corners as an integer polygon, ready for use with
    cv2.fillPoly / cv2.pointPolygonTest for masking.

    This convex hull is the full "could a piece on this square possibly
    appear here" region — it's normally a hexagon-ish or quad-ish shape
    that leans toward the camera at the top, matching how a real piece's
    silhouette would lean in the image due to perspective.
    """
    corners_3d = square_prism_3d(file_idx, rank_idx, height_mm, shrink_mm)
    corners_2d = project_points(corners_3d, rvec, tvec, camera_matrix, dist_coeffs)
    hull = cv2.convexHull(corners_2d.astype(np.float32))
    return hull.reshape(-1, 2).astype(np.int32)


def square_height_slice_image_quad(file_idx, rank_idx, z_mm, rvec, tvec,
                                    camera_matrix, dist_coeffs, shrink_mm=3.0):
    """
    Project just the footprint of a square at a SPECIFIC height z_mm
    (a horizontal slice through the prism) into the image, returning a
    quad. Used by occupancy.py to sample multiple height slices and
    confirm a detected blob is actually anchored to the right square
    at a plausible piece height, rather than relying on the full prism
    hull alone (see occupancy.py docstring for why both matter).
    """
    base = square_footprint_3d(file_idx, rank_idx, shrink_mm)
    slice_pts = base.copy()
    slice_pts[:, 2] = z_mm
    pts_2d = project_points(slice_pts, rvec, tvec, camera_matrix, dist_coeffs)
    return pts_2d.astype(np.int32)


def all_square_prism_hulls(rvec, tvec, camera_matrix, dist_coeffs,
                            height_mm=PIECE_MAX_HEIGHT_MM, shrink_mm=3.0):
    """
    Convenience: compute prism image hulls for all 64 squares at once.
    Returns dict: "a1".."h8" -> Nx2 int32 polygon.
    """
    result = {}
    for r in range(8):
        for f in range(8):
            name = f"{FILES[f]}{RANKS[r]}"
            result[name] = square_prism_image_hull(
                f, r, rvec, tvec, camera_matrix, dist_coeffs,
                height_mm, shrink_mm
            )
    return result
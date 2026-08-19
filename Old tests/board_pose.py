"""
board_pose.py

Detects 4 corner AprilTags on the chess board and solves for the camera's
pose (rotation + translation) relative to the board plane.

ASSUMED SETUP (adjust constants below to match your physical rig):
- 4 AprilTags, one near each corner of the 8x8 playing area, all lying flat
  on the board plane (z=0 in board coordinates).
- Tag family: DICT_APRILTAG_36h11 (change if yours differs).
- You know:
    - the side length of the playing area (8 squares) in mm
    - the tag IDs assigned to each corner
    - the tag size (mm) and how far each tag's center is inset from the
      actual board corner (since tags usually sit just outside or just
      inside the playing area, not exactly on the corner point).

Coordinate system (board frame):
    Origin = corner of square a1 (outermost corner, i.e. the actual
             corner of the 8x8 grid), x-axis along the a-h files,
             y-axis along the 1-8 ranks, z-axis pointing UP out of the board.

You will likely need to tweak TAG_BOARD_POSITIONS to match your physical
tag placement. Print the tags, measure once with calipers/ruler, done.
"""

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# PHYSICAL CONSTANTS — edit these to match your board
# ---------------------------------------------------------------------------

SQUARE_SIZE_MM = 35.0          # edge length of one chess square
BOARD_SIZE_MM = SQUARE_SIZE_MM * 8.0

TAG_FAMILY = cv2.aruco.DICT_APRILTAG_36h11
TAG_SIZE_MM = 20.0             # edge length of the black square of the tag itself

# Tag IDs at each corner of the board (you choose these when printing tags)
# Mapping: tag_id -> (x_mm, y_mm) of the TAG CENTER in board coordinates.
# Board coordinates: origin at a1's outer corner, x toward h-file, y toward
# rank 8, z up. Adjust the offsets (here: 15mm outside the grid on each
# axis) to match where you actually stuck the tags.
TAG_INSET_MM = 15.0  # how far outside the 8x8 grid the tag centers sit

TAG_BOARD_POSITIONS = {
    0: (-TAG_INSET_MM,            -TAG_INSET_MM),              # near a1 corner
    1: (BOARD_SIZE_MM + TAG_INSET_MM, -TAG_INSET_MM),          # near h1 corner
    2: (BOARD_SIZE_MM + TAG_INSET_MM, BOARD_SIZE_MM + TAG_INSET_MM),  # near h8
    3: (-TAG_INSET_MM,            BOARD_SIZE_MM + TAG_INSET_MM),     # near a8
}

# ---------------------------------------------------------------------------


def _tag_corners_3d(center_xy, size_mm):
    """
    Return the 4 corner points of a tag in board-frame 3D coordinates,
    given its center (x, y) and edge size. Order matches the order OpenCV's
    ArUco detector returns corners in: top-left, top-right, bottom-right,
    bottom-left (in the *tag's own* image-space sense). We define them
    consistently here; what matters is consistency, not which physical
    corner is "first", since solvePnP just needs matched correspondences.
    """
    cx, cy = center_xy
    h = size_mm / 2.0
    # NOTE: tags lie flat on the board, so z = 0 for all corners.
    # We order them: (-x,+y), (+x,+y), (+x,-y), (-x,-y) which is the
    # typical "top-left, top-right, bottom-right, bottom-left" convention
    # when y increases "up" in image-like terms. If your detected pose
    # looks mirrored/rotated, this ordering is the first thing to flip.
    return np.array([
        [cx - h, cy + h, 0.0],
        [cx + h, cy + h, 0.0],
        [cx + h, cy - h, 0.0],
        [cx - h, cy - h, 0.0],
    ], dtype=np.float64)


class BoardPoseEstimator:
    def __init__(self, camera_matrix, dist_coeffs,
                 tag_family=TAG_FAMILY,
                 tag_positions=TAG_BOARD_POSITIONS,
                 tag_size_mm=TAG_SIZE_MM):
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.tag_positions = tag_positions
        self.tag_size_mm = tag_size_mm

        self.aruco_dict = cv2.aruco.getPredefinedDictionary(tag_family)
        self.aruco_params = cv2.aruco.DetectorParameters()
        # AprilTag-specific corner refinement gives noticeably better pose
        # accuracy than the default, which matters a lot for far-side
        # squares where reprojection error gets amplified by distance.
        self.aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)

    def detect_tags(self, gray_image):
        """Returns (corners_list, ids) from the raw ArUco/AprilTag detector."""
        corners, ids, _rejected = self.detector.detectMarkers(gray_image)
        return corners, ids

    def estimate_pose(self, gray_image, min_tags=3):
        """
        Detect corner tags and solve for camera pose relative to the board.

        Returns:
            success (bool)
            rvec, tvec: board-to-camera rotation/translation (None if failed)
            debug_info: dict with detected tag ids and reprojection error
        """
        corners, ids = self.detect_tags(gray_image)

        if ids is None:
            return False, None, None, {"reason": "no_tags_detected"}

        ids = ids.flatten()

        object_points = []
        image_points = []
        used_ids = []

        for tag_corners, tag_id in zip(corners, ids):
            if tag_id not in self.tag_positions:
                continue  # ignore tags that aren't part of the board rig
            obj_pts = _tag_corners_3d(self.tag_positions[tag_id], self.tag_size_mm)
            img_pts = tag_corners.reshape(4, 2)
            object_points.append(obj_pts)
            image_points.append(img_pts)
            used_ids.append(int(tag_id))

        if len(used_ids) < min_tags:
            return False, None, None, {
                "reason": "not_enough_known_tags",
                "found_ids": [int(i) for i in ids],
            }

        object_points = np.vstack(object_points)
        image_points = np.vstack(image_points)

        # SOLVEPNP_ITERATIVE with all 4 tags' worth of corners (16 points
        # if all 4 tags seen) is well-conditioned and accurate. If you
        # only see 3 tags it still works, just slightly less robust.
        success, rvec, tvec = cv2.solvePnP(
            object_points, image_points,
            self.camera_matrix, self.dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not success:
            return False, None, None, {"reason": "solvepnp_failed"}

        # Reprojection error as a sanity-check / confidence signal.
        # Re-run pose with refinement using all points, then measure.
        rvec, tvec = cv2.solvePnPRefineLM(
            object_points, image_points, self.camera_matrix, self.dist_coeffs,
            rvec, tvec
        )
        projected, _ = cv2.projectPoints(
            object_points, rvec, tvec, self.camera_matrix, self.dist_coeffs
        )
        err = np.linalg.norm(projected.reshape(-1, 2) - image_points, axis=1).mean()

        return True, rvec, tvec, {
            "used_ids": used_ids,
            "mean_reprojection_error_px": float(err),
        }
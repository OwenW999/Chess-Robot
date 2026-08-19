import cv2
import numpy as np

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_16h5)
detector_params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, detector_params)

cap = cv2.VideoCapture(0)

# Map tag ID -> role
TAG_ROLES = {0: "top-left", 1: "top-right", 2: "bottom-right", 3: "bottom-left"}

# Your AprilTags sit OUTSIDE the playable 8x8 area (deliberately, so pieces
# don't cover them). That means the tag CENTERS are not the actual board
# corners -- they're offset outward by some amount. We need to shrink the
# quadrilateral inward before warping, or every square in our grid will be
# misaligned with the real board squares.
#
# Set this to how far (in real-world units, e.g. mm) each tag's center is
# from the actual nearest board corner, MEASURED ALONG THE BOARD SURFACE,
# in the same direction as the board edge (not diagonally to the corner).
# Easiest way to get this: measure it with a ruler on your physical setup.
TAG_OFFSET_MM = 30.0      # <-- measure this on your rig and update it
SQUARE_SIZE_MM = 49.0     # <-- real size of one chess square on your board, measure it
BOARD_SIZE_MM = SQUARE_SIZE_MM * 8.0
# Tag-to-tag spacing along one edge = true board edge + offset on both ends.
TAG_SPACING_MM = BOARD_SIZE_MM + 2 * TAG_OFFSET_MM
# Fraction of the tag-to-tag distance to move inward from each tag center
# to land exactly on the true board corner. This must be computed relative
# to TAG-TO-TAG distance (what shrink_corner's direction vectors actually
# measure), not the true board size -- using the wrong denominator here
# was a bug caught only by checking the numbers, not by eye.
TAG_OFFSET_FRACTION = TAG_OFFSET_MM / TAG_SPACING_MM

# ---------------------------------------------------------------------------
# Occupancy detection config
# ---------------------------------------------------------------------------
WARP_SIZE = 800              # matches your existing warp size
SQUARE_PX = WARP_SIZE // 8

FILES = "abcdefgh"
RANKS = "12345678"

# IMPORTANT -- read this before trusting results:
# Your pts order is [top-left, top-right, bottom-right, bottom-left], mapped
# to dst corners [0,0], [size,0], [size,size], [0,size]. That means:
#   y = 0   -> your "top" tags
#   y = size -> your "bottom" tags
# The ghost-bleed math below assumes y increases TOWARD the camera (i.e.
# your "bottom" tags are the near-camera edge of the board, "top" tags are
# the far edge). If your physical rig has it the other way around (camera
# looking from the "top" tag side), flip SAFE_EDGE_IS_TOP_OF_SQUARE to False
# below, or just swap your TAG_ROLES naming -- either fixes it.
SAFE_EDGE_IS_TOP_OF_SQUARE = True

# Fraction of each square's height we sample for occupancy. This is the
# strip furthest from the camera-side neighbor, which avoids being fooled
# by a tall piece (e.g. queen) in the square in front of this one bleeding
# its "ghost" into this square during the warp. Tune once you see real
# footage -- start at 0.4, go smaller if you still get false positives on
# squares behind tall pieces, go larger if real pieces aren't triggering
# strongly enough.
SAFE_EDGE_FRACTION = 0.45


def square_to_warped_rect(square):
    """
    Map a square name like 'e4' to its (x, y, w, h) box in the warped image.
    rank 8 is the far row (y=0), rank 1 is the near row (y=large) -- matches
    a standard board setup with rank 8 on the far/top side away from the
    player operating the camera. Swap if your physical setup differs.
    """
    file_idx = FILES.index(square[0])
    rank_idx = RANKS.index(square[1])
    col = file_idx
    row = 7 - rank_idx
    x = col * SQUARE_PX
    y = row * SQUARE_PX
    return (x, y, SQUARE_PX, SQUARE_PX)


def safe_edge_crop_rect(square_rect):
    """Return the sub-rect of a square that's safest from neighbor-ghost bleed."""
    x, y, w, h = square_rect
    safe_h = int(h * SAFE_EDGE_FRACTION)
    if SAFE_EDGE_IS_TOP_OF_SQUARE:
        return (x, y, w, safe_h)               # top strip (far from camera)
    else:
        return (x, y + (h - safe_h), w, safe_h)  # bottom strip, if your rig is flipped


ALL_SQUARES = [f + r for r in RANKS for f in FILES]


def square_color(square):
    """
    Standard chess coloring: a1 is dark. (file_idx + rank_idx) even -> dark,
    odd -> light, using 0-indexed file/rank. Used to calibrate separately
    per square color, since dark and light wood/material can have very
    different inherent texture/grain.
    """
    file_idx = FILES.index(square[0])
    rank_idx = RANKS.index(square[1])
    return "dark" if (file_idx + rank_idx) % 2 == 0 else "light"


# Squares on ranks 3-6 are guaranteed empty in a standard starting position,
# regardless of color. We use these to calibrate a per-color baseline score
# the first time we see a fully-set-up starting position, instead of
# requiring a separate empty-board photo step.
CALIBRATION_RANKS = {"3", "4", "5", "6"}
CALIBRATION_SQUARES = [sq for sq in ALL_SQUARES if sq[1] in CALIBRATION_RANKS]

# Populated once at calibration time: {"dark": (mean, std), "light": (mean, std)}
color_baseline = {}
is_calibrated = False

# How many standard deviations above a color's own empty-square baseline
# counts as "occupied." Edge-density scores are a much cleaner signal than
# raw pixel std() was (shadows and lighting drift barely move this number,
# see the comment on raw_score below), so this threshold can likely sit a
# bit higher/cleaner than the old one once you see real numbers. Press 'd'
# on your real board (empty AND with pieces, both square colors) and look
# at the actual sigma gap before trusting this value.
OCCUPANCY_SIGMA_THRESHOLD = 2.25


def raw_score(gray_warped, square):
    """
    The occupancy signal for one square's safe-edge crop.

    CHANGED from a plain pixel std() to a local-contrast-normalized
    gradient (edge) density:

      - std() measures ANY brightness variation in the crop, including
        soft shadows and lighting gradients, which inflated scores on
        empty squares with a shadow falling across them -- shadows have
        no hard edges, just a smooth brightness ramp, but std() doesn't
        distinguish "smooth ramp" from "real texture," it just sees
        numbers spreading out either way.
      - This version divides the crop by a heavily-blurred version of
        itself first. That cancels out anything that varies SLOWLY across
        the crop (smooth shadow falloff, uneven lighting, slow vignetting)
        while preserving anything that varies QUICKLY (real edges: a
        piece's silhouette boundary, facets, base rim, engraving).
      - Sobel gradients are then computed on that normalized crop and
        averaged. A flat or shadowed square -> low average gradient.
        A square with a piece's edge in the safe-edge strip -> high
        average gradient, regardless of how dark/light the shadow or
        ambient lighting happens to be that frame.

    This is the fix for "inconsistent and random" occupancy flips: the
    old signal was sensitive to exactly the things that drift frame-to-
    frame (auto white balance, exposure, soft shadows), and not
    specifically sensitive to the thing that actually matters (is there
    a 3D object's edge here).
    """
    full_rect = square_to_warped_rect(square)
    cx, cy, cw, ch = safe_edge_crop_rect(full_rect)
    crop = gray_warped[cy:cy + ch, cx:cx + cw]

    if crop.size == 0:
        return 0.0

    crop_f = crop.astype(np.float32) + 1.0
    # Blur kernel must be smaller than the crop itself or GaussianBlur
    # degenerates -- clamp to an odd size no bigger than ~half the crop.
    max_k = max(3, (min(crop.shape) // 2) | 1)  # force odd
    k = min(31, max_k)
    if k % 2 == 0:
        k -= 1
    k = max(3, k)

    local_bg = cv2.GaussianBlur(crop_f, (k, k), 0) + 1.0
    normalized = crop_f / local_bg  # ~1.0 in flat/shadowed regions

    grad_x = cv2.Sobel(normalized, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(normalized, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(grad_x, grad_y)

    # Scale up for friendlier numbers in print/debug output -- purely
    # cosmetic, doesn't change relative comparisons or sigma math.
    return float(grad_mag.mean()) * 1000.0


def calibrate_from_starting_position(gray_warped):
    """
    Call this once, when the user confirms the board is set up in the
    standard starting position. Computes a per-color baseline (mean +
    std of the empty-square score) from the guaranteed-empty middle ranks.
    """
    global color_baseline, is_calibrated
    by_color = {"dark": [], "light": []}
    for sq in CALIBRATION_SQUARES:
        by_color[square_color(sq)].append(raw_score(gray_warped, sq))

    color_baseline = {
        color: (float(np.mean(scores)), float(np.std(scores)) + 1e-6)
        for color, scores in by_color.items()
    }
    is_calibrated = True
    print("Calibrated baselines:", color_baseline)


def compute_occupancy(warped_bgr):
    """
    Returns a dict: square name -> (is_occupied: bool, score: float, sigma: float)

    Occupancy is relative to each square's OWN COLOR's empty baseline,
    using the edge-density score from raw_score() above.
    """
    gray = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY)
    results = {}
    for sq in ALL_SQUARES:
        score = raw_score(gray, sq)
        if is_calibrated:
            mean, std = color_baseline[square_color(sq)]
            sigma = (score - mean) / std
            is_occupied = sigma > OCCUPANCY_SIGMA_THRESHOLD
        else:
            # Not calibrated yet -- can't make a reliable call. Report
            # everything as not-occupied rather than guessing with a
            # fixed threshold.
            sigma = 0.0
            is_occupied = False
        results[sq] = (is_occupied, score, sigma)
    return results


def draw_occupancy_overlay(warped_bgr, occupancy):
    """
    Draw the full square grid, the safe-edge sample region, and a clear
    occupied/empty indicator (green filled circle = occupied) on top of
    the warped board image.
    """
    out = warped_bgr.copy()
    for sq in ALL_SQUARES:
        full_rect = square_to_warped_rect(sq)
        fx, fy, fw, fh = full_rect
        cx, cy, cw, ch = safe_edge_crop_rect(full_rect)

        is_occupied, score, sigma = occupancy[sq]

        # Square grid outline (dim gray)
        cv2.rectangle(out, (fx, fy), (fx + fw, fy + fh), (90, 90, 90), 1)

        # Safe-edge sample region outline (yellow)
        cv2.rectangle(out, (cx, cy), (cx + cw, cy + ch), (0, 200, 255), 1)

        center = (fx + fw // 2, fy + fh // 2)
        if is_occupied:
            cv2.circle(out, center, 14, (0, 255, 0), -1)   # filled green = occupied
        else:
            cv2.circle(out, center, 14, (60, 60, 60), 1)   # thin gray outline = empty

        # Square label, small, top-left of each cell
        cv2.putText(out, sq, (fx + 4, fy + 14), cv2.FONT_HERSHEY_SIMPLEX,
                    0.35, (255, 255, 255), 1, cv2.LINE_AA)

    if not is_calibrated:
        cv2.putText(out, "NOT CALIBRATED -- set up starting position, press 'c'",
                    (10, out.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 2, cv2.LINE_AA)

    return out


# ---------------------------------------------------------------------------
# Main loop (your original structure, with occupancy detection added)
# ---------------------------------------------------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    corners, ids, rejected = detector.detectMarkers(frame)

    display = frame.copy()
    board_corners = {}

    if ids is not None:
        cv2.aruco.drawDetectedMarkers(display, corners, ids)

        for i, tag_id in enumerate(ids.flatten()):
            if tag_id in TAG_ROLES:
                tag_corners = corners[i][0]
                center = tag_corners.mean(axis=0)
                board_corners[TAG_ROLES[tag_id]] = center

    if len(board_corners) == 4:
        pts = np.array([
            board_corners["top-left"],
            board_corners["top-right"],
            board_corners["bottom-right"],
            board_corners["bottom-left"]
        ], dtype="float32")

        cv2.polylines(display, [pts.astype(int)], isClosed=True, color=(0, 255, 0), thickness=3)

        # Shrink inward from tag centers to the actual board corners.
        tl, tr, br, bl = pts[0], pts[1], pts[2], pts[3]

        def shrink_corner(corner, along_edge_a, along_edge_b, frac):
            dir_a = (along_edge_a - corner)
            dir_b = (along_edge_b - corner)
            return corner + dir_a * frac + dir_b * frac

        corrected_tl = shrink_corner(tl, tr, bl, TAG_OFFSET_FRACTION)
        corrected_tr = shrink_corner(tr, tl, br, TAG_OFFSET_FRACTION)
        corrected_br = shrink_corner(br, bl, tr, TAG_OFFSET_FRACTION)
        corrected_bl = shrink_corner(bl, br, tl, TAG_OFFSET_FRACTION)

        pts_corrected = np.array([corrected_tl, corrected_tr, corrected_br, corrected_bl],
                                  dtype="float32")

        cv2.polylines(display, [pts_corrected.astype(int)], isClosed=True,
                      color=(255, 0, 255), thickness=2)  # magenta = corrected board outline

        size = WARP_SIZE
        dst = np.array([[0, 0], [size, 0], [size, size], [0, size]], dtype="float32")
        matrix = cv2.getPerspectiveTransform(pts_corrected, dst)
        warped = cv2.warpPerspective(frame, matrix, (size, size))

        occupancy = compute_occupancy(warped)
        warped_overlay = draw_occupancy_overlay(warped, occupancy)

        cv2.imshow("Warped Board", warped_overlay)

        all_scores = [score for _, score, _ in occupancy.values()]
        status = "CALIBRATED" if is_calibrated else "NOT CALIBRATED (press 'c')"
        print(f"[{status}] edge score min={min(all_scores):.2f} "
              f"max={max(all_scores):.2f} "
              f"mean={sum(all_scores)/len(all_scores):.2f}",
              end="\r")

    cv2.imshow("Detection", display)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == ord('c') and len(board_corners) == 4:
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        calibrate_from_starting_position(gray)
    if key == ord('d') and len(board_corners) == 4:
        print("\n--- per-square scores (sorted by sigma) ---")
        for sq, (occ, score, sigma) in sorted(occupancy.items(), key=lambda kv: kv[1][2]):
            print(f"{sq} ({square_color(sq)}): edge_score={score:.2f} sigma={sigma:.2f}  "
                  f"{'OCCUPIED' if occ else ''}")
        print("--- end ---\n")

cap.release()
cv2.destroyAllWindows()
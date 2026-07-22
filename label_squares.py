"""
label_squares.py

Interactive tool for labeling chessboard squares as empty / white / black,
to build a training set for the per-square occupancy+color CNN.

WORKFLOW:
  1. Point this at a folder of captured frames (full chessboard images).
  2. For each frame: AprilTag-based homography warps it to a top-down
     view, then this script slices it into a 64-square grid and
     overlays clickable cells.
  3. Click a square to cycle its state: empty -> white -> black -> empty.
     - empty:  no highlight
     - white:  blue highlight (piece belongs to white)
     - black:  red highlight (piece belongs to black)
     Hold 'b' while clicking to force a square straight to black,
     instead of cycling through it -- handy for quickly labeling a run
     of black pieces without extra clicks.
  4. Press 's' to crop and write out all 64 squares into
     dataset/empty/, dataset/white/, dataset/black/, then auto-advances
     to the next frame.

Controls:
  Left click on a square       : cycle empty -> white -> black -> empty
  Hold 'b' + left click        : force that square to black
  's'                          : save labels for this frame + advance
  'n'                          : skip frame without saving
  'r'                          : reset all toggles on current frame to empty
  'q'                          : quit
"""

import cv2
import numpy as np
import os
import glob
import json
import re
import time
import argparse

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_16h5)
detector_params = cv2.aruco.DetectorParameters()

# detector_params.adaptiveThreshWinSizeMin = 3
# detector_params.adaptiveThreshWinSizeMax = 30
# detector_params.adaptiveThreshWinSizeStep = 2
# detector_params.adaptiveThreshConstant = 7
# detector_params.minMarkerPerimeterRate = 0.01
# detector_params.maxMarkerPerimeterRate = 5.0
# detector_params.polygonalApproxAccuracyRate = 0.09

# Wider/finer adaptive threshold search — catches more lighting conditions
detector_params.adaptiveThreshWinSizeMin = 3
detector_params.adaptiveThreshWinSizeMax = 75
detector_params.adaptiveThreshWinSizeStep = 4
detector_params.adaptiveThreshConstant = 7

# Let smaller/farther tags count as candidates
detector_params.minMarkerPerimeterRate = 0.01
detector_params.maxMarkerPerimeterRate = 6.0
detector_params.polygonalApproxAccuracyRate = 0.14

# Looser polygon fit tolerates blur/motion/warped corners
detector_params.polygonalApproxAccuracyRate = 0.10

detector_params.errorCorrectionRate = 0.9

detector_params.minCornerDistanceRate = 0.02
detector_params.minDistanceToBorder = 0
detector_params.minMarkerDistanceRate = 0.02

detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
detector_params.cornerRefinementWinSize = 7
detector_params.cornerRefinementMaxIterations = 50
detector_params.cornerRefinementMinAccuracy = 0.05

detector_params.perspectiveRemovePixelPerCell = 8
detector_params.perspectiveRemoveIgnoredMarginPerCell = 0.20

detector = cv2.aruco.ArucoDetector(aruco_dict, detector_params)

# NOTE: no cv2.VideoCapture here -- this script only ever processes saved
# image files from INPUT_FRAMES_DIR. It must never touch a live camera;
# that was the bug (warp_board was secretly re-reading from a live cap
# instead of using the loaded frame, so every frame looked "the same
# but different brightness" -- it WAS a different, live frame each
# time, just not the one you thought you were labeling).

# Map tag ID -> role
TAG_ROLES = {0: "top-left", 1: "top-right", 2: "bottom-right", 3: "bottom-left"}

# ----------------------------- CONFIG -----------------------------

INPUT_FRAMES_DIR = "./raw_frames"          # folder of full-board images to label
OUTPUT_DATASET_DIR = "./dataset"           # where labeled crops get saved
BOARD_SIZE = 800                            # size (px) of the warped top-down board
SQUARE_SIZE = BOARD_SIZE // 8                # px per square in the warped image
CROP_PADDING = 4                             # extra px around each square crop
MANIFEST_PATH = os.path.join(OUTPUT_DATASET_DIR, "manifest.jsonl")

TAG_OFFSET_MM = 30.5      # <-- measure this on your rig and update it
SQUARE_SIZE_MM = 47       # <-- real size of one chess square on your board, measure it
BOARD_SIZE_MM = SQUARE_SIZE_MM * 8.0
TAG_SPACING_MM = BOARD_SIZE_MM + 2 * TAG_OFFSET_MM
TAG_OFFSET_FRACTION = TAG_OFFSET_MM / TAG_SPACING_MM

WARP_SIZE = 800
SQUARE_PX = WARP_SIZE // 8

FILES = "abcdefgh"
RANKS = "12345678"

# Square state constants -- shared with train/test scripts by convention
# (0 = empty, 1 = white piece, 2 = black piece). Cycling order below
# follows this same sequence.
STATE_EMPTY = 0
STATE_WHITE = 1
STATE_BLACK = 2
STATE_NAMES = {STATE_EMPTY: "empty", STATE_WHITE: "white", STATE_BLACK: "black"}

# Overlay colors (BGR) per state. Empty gets no overlay.
STATE_OVERLAY_COLOR = {
    STATE_WHITE: (255, 120, 0),   # blue-ish highlight = white piece
    STATE_BLACK: (0, 0, 255),     # red highlight = black piece
}

# OpenCV's waitKey only reports key-DOWN events, not key-up, so there's no
# direct way to ask "is 'b' currently held". Instead we treat 'b' as "still
# held" if we saw it fire (via the OS's own keyboard auto-repeat) within
# this many seconds. Comfortably longer than one waitKey(20) poll interval,
# short enough that releasing the key stops the override almost immediately.
B_HOLD_TIMEOUT_SEC = 0.35

# ----------------------------- HOMOGRAPHY -----------------------------

def warp_board(raw_frame):
    """
    Takes a single still image (already loaded, e.g. via cv2.imread) and
    returns a top-down warped board image of shape (WARP_SIZE, WARP_SIZE, 3),
    or None if all 4 AprilTags weren't found.

    IMPORTANT: this must operate ONLY on raw_frame, the argument passed in.
    No live camera access here -- this fn needs to be deterministic given
    a fixed input image, since it's called once per saved file when
    labeling, and must reproduce the same warp every time for the same file.
    """
    frame = raw_frame

    corners, ids, rejected = detector.detectMarkers(frame)
    board_corners = {}

    if ids is not None:
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

        size = WARP_SIZE
        dst = np.array([[0, 0], [size, 0], [size, size], [0, size]], dtype="float32")
        matrix = cv2.getPerspectiveTransform(pts, dst)
        warped_full = cv2.warpPerspective(frame, matrix, (size, size))

        margin = int(size * TAG_OFFSET_FRACTION)
        warped = warped_full[margin:size - margin, margin:size - margin]
        warped = cv2.resize(warped, (size, size))
        return warped

    print("Could not find all 4 tags. Make sure the board is fully visible and tags are not occluded.")
    return None


# ----------------------------- LABELING UI -----------------------------

class SquareLabeler:
    def __init__(self, frame_paths):
        self.frame_paths = frame_paths
        self.frame_idx = 0
        # state[row][col] in {STATE_EMPTY, STATE_WHITE, STATE_BLACK}
        self.state = [[STATE_EMPTY] * 8 for _ in range(8)]
        self.window_name = (
            "Label squares - click cycles empty/white/black (hold 'b' = force black), "
            "'s' save, 'n' skip, 'r' reset, 'q' quit"
        )
        self.warped = None
        # Timestamp of the most recent 'b' keypress seen in run()'s loop.
        # None means 'b' is not currently considered held.
        self._b_last_seen = None

    def _init_window(self):
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.on_click)

    def _is_b_held(self):
        return (self._b_last_seen is not None
                and (time.time() - self._b_last_seen) < B_HOLD_TIMEOUT_SEC)

    def note_key(self, key):
        """Called once per polled key in run()'s loop to update hold-state tracking."""
        if key == ord('b'):
            self._b_last_seen = time.time()

    def on_click(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        col = x // SQUARE_SIZE
        row = y // SQUARE_SIZE
        if 0 <= row < 8 and 0 <= col < 8:
            if self._is_b_held():
                # Force straight to black instead of cycling -- lets you
                # rapid-click a run of black pieces without extra clicks.
                self.state[row][col] = STATE_BLACK
            else:
                # Cycle empty -> white -> black -> empty
                self.state[row][col] = (self.state[row][col] + 1) % 3

    def draw_overlay(self):
        display = self.warped.copy()
        for row in range(8):
            for col in range(8):
                x0, y0 = col * SQUARE_SIZE, row * SQUARE_SIZE
                x1, y1 = x0 + SQUARE_SIZE, y0 + SQUARE_SIZE
                cv2.rectangle(display, (x0, y0), (x1, y1), (60, 60, 60), 1)
                state = self.state[row][col]
                if state != STATE_EMPTY:
                    overlay = display.copy()
                    cv2.rectangle(overlay, (x0, y0), (x1, y1),
                                  STATE_OVERLAY_COLOR[state], -1)
                    display = cv2.addWeighted(overlay, 0.35, display, 0.65, 0)
        progress = f"Frame {self.frame_idx + 1}/{len(self.frame_paths)}"
        cv2.putText(display, progress, (10, BOARD_SIZE - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        legend = "blue=white  red=black  none=empty  |  hold 'b'+click = force black"
        cv2.putText(display, legend, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        return display

    def reset_toggles(self):
        self.state = [[STATE_EMPTY] * 8 for _ in range(8)]

    def _remove_stale_crop(self, frame_name, row, col):
        """
        Deletes any existing crop for this exact square from ALL label
        folders, regardless of which class it was previously saved under.

        This matters for relabeling: if a square was labeled 'white' in an
        earlier session and you now relabel the same frame as 'black', a
        naive save would leave the old dataset/white/{frame}_r{row}c{col}.png
        sitting around too -- same square, two contradictory copies, both
        picked up by build_datasets(). Removing any prior copy first keeps
        exactly one crop per square on disk at all times.
        """
        pattern = f"{frame_name}_r{row}c{col}.*"
        for label_dir in STATE_NAMES.values():
            for stale_path in glob.glob(os.path.join(OUTPUT_DATASET_DIR, label_dir, pattern)):
                os.remove(stale_path)

    def save_current(self):
        for label_dir in STATE_NAMES.values():
            os.makedirs(os.path.join(OUTPUT_DATASET_DIR, label_dir), exist_ok=True)

        frame_name = os.path.splitext(os.path.basename(self.frame_paths[self.frame_idx]))[0]
        manifest_entries = []
        counts = {STATE_EMPTY: 0, STATE_WHITE: 0, STATE_BLACK: 0}

        for row in range(8):
            for col in range(8):
                x0 = max(col * SQUARE_SIZE - CROP_PADDING, 0)
                y0 = max(row * SQUARE_SIZE - CROP_PADDING, 0)
                x1 = min(x0 + SQUARE_SIZE + 2 * CROP_PADDING, BOARD_SIZE)
                y1 = min(y0 + SQUARE_SIZE + 2 * CROP_PADDING, BOARD_SIZE)
                crop = self.warped[y0:y1, x0:x1]

                state = self.state[row][col]
                label = STATE_NAMES[state]
                counts[state] += 1

                self._remove_stale_crop(frame_name, row, col)

                out_name = f"{frame_name}_r{row}c{col}.png"
                out_path = os.path.join(OUTPUT_DATASET_DIR, label, out_name)
                cv2.imwrite(out_path, crop)

                manifest_entries.append({
                    "frame": frame_name, "row": row, "col": col,
                    "label": label, "path": out_path
                })

        with open(MANIFEST_PATH, "a") as f:
            for entry in manifest_entries:
                f.write(json.dumps(entry) + "\n")

        print(f"[saved] {frame_name}: "
              f"{counts[STATE_EMPTY]} empty / "
              f"{counts[STATE_WHITE]} white / "
              f"{counts[STATE_BLACK]} black")

    def run(self):
        self._init_window()
        while self.frame_idx < len(self.frame_paths):
            raw = cv2.imread(self.frame_paths[self.frame_idx])
            if raw is None:
                print(f"[skip] couldn't read {self.frame_paths[self.frame_idx]}")
                self.frame_idx += 1
                continue

            self.warped = warp_board(raw)
            if self.warped is None:
                print(f"[skip] frame {self.frame_idx + 1} - tags not detected")
                self.frame_idx += 1
                continue
            self.reset_toggles()

            while True:
                display = self.draw_overlay()
                cv2.imshow(self.window_name, display)
                key = cv2.waitKey(20) & 0xFF
                self.note_key(key)

                if key == ord('s'):
                    self.save_current()
                    self.frame_idx += 1
                    break
                elif key == ord('n'):
                    print(f"[skip, no save] frame {self.frame_idx + 1}")
                    self.frame_idx += 1
                    break
                elif key == ord('r'):
                    self.reset_toggles()
                elif key == ord('q'):
                    cv2.destroyAllWindows()
                    return

        cv2.destroyAllWindows()
        print("All frames labeled.")


def get_already_labeled_frames():
    """
    Returns the set of frame names that have already been fully saved by
    a previous labeling session, so main() can skip them.

    Detection strategy: save_current() writes one crop per square named
    '{frame_name}_r{row}c{col}.{ext}' into dataset/empty/, dataset/white/,
    or dataset/black/. We only need to check ONE square (r0c0) per frame --
    save_current() always writes all 64 in a single call, so if r0c0
    exists in ANY of the three label folders, the whole frame was saved.
    This avoids needing a separate 'completed frames' tracking file that
    could drift out of sync with what's actually on disk.
    """
    labeled = set()
    for label_dir in STATE_NAMES.values():
        for path in glob.glob(os.path.join(OUTPUT_DATASET_DIR, label_dir, "*_r0c0.*")):
            base = os.path.basename(path)
            match = re.match(r"(.+)_r0c0\.\w+$", base)
            if match:
                labeled.add(match.group(1))
    return labeled


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--relabel-all", action="store_true",
        help="Re-label every frame in raw_frames/, including ones already "
             "saved in a previous session (normally those are skipped). "
             "Existing crops for a re-labeled square are replaced, not "
             "duplicated, even if you change its class."
    )
    args = parser.parse_args()

    frame_paths = sorted(glob.glob(os.path.join(INPUT_FRAMES_DIR, "*.jpg")) +
                          glob.glob(os.path.join(INPUT_FRAMES_DIR, "*.png")))
    if not frame_paths:
        print(f"No frames found in {INPUT_FRAMES_DIR}. "
              f"Capture some board photos there first (jpg or png).")
        return

    if args.relabel_all:
        remaining = frame_paths
        print(f"--relabel-all set: relabeling all {len(frame_paths)} frames "
              f"(ignoring any previous labels).")
    else:
        already_labeled = get_already_labeled_frames()
        remaining = [p for p in frame_paths
                     if os.path.splitext(os.path.basename(p))[0] not in already_labeled]
        skipped_count = len(frame_paths) - len(remaining)
        print(f"Found {len(frame_paths)} frames total "
              f"({skipped_count} already labeled, {len(remaining)} remaining).")

    if not remaining:
        print("Nothing left to label -- every frame in raw_frames/ already has a saved entry in dataset/.")
        return

    labeler = SquareLabeler(remaining)
    labeler.run()


if __name__ == "__main__":
    main()
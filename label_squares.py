# # """
# # label_squares.py

# # Interactive tool for labeling chessboard squares as occupied/empty,
# # to build a training set for the per-square occupancy CNN.

# # WORKFLOW:
# #   1. Point this at a folder of captured frames (full chessboard images),
# #      or hook up a live camera (see CAPTURE SOURCE section below).
# #   2. For each frame: your existing homography pipeline warps it to a
# #      top-down view, then this script slices it into a 64-square grid
# #      and overlays clickable cells.
# #   3. Click a square to toggle occupied (red) / empty (no highlight).
# #   4. Press SAVE to crop and write out all 64 squares into
# #      dataset/occupied/ and dataset/empty/, then auto-advances to the
# #      next frame.

# # YOU NEED TO PLUG IN: your existing homography + corner-detection code
# # in `warp_board()` below. I've stubbed it with a placeholder that
# # assumes you already have a function that returns a clean top-down
# # board image (a square image, ideally a fixed size like 800x800) given
# # a raw frame. Swap that call out for your AprilTag homography code.

# # Controls:
# #   Left click on a square : toggle occupied/empty
# #   's'                    : save labels for this frame + advance
# #   'n'                    : skip frame without saving
# #   'r'                    : reset all toggles on current frame to empty
# #   'q'                    : quit
# # """

# # import cv2
# # import numpy as np
# # import os
# # import glob
# # import json

# # aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_16h5)
# # detector_params = cv2.aruco.DetectorParameters()
# # detector = cv2.aruco.ArucoDetector(aruco_dict, detector_params)

# # cap = cv2.VideoCapture(0)

# # # Map tag ID -> role
# # TAG_ROLES = {0: "top-left", 1: "top-right", 2: "bottom-right", 3: "bottom-left"}

# # # ----------------------------- CONFIG -----------------------------

# # INPUT_FRAMES_DIR = "./raw_frames"          # folder of full-board images to label
# # OUTPUT_DATASET_DIR = "./dataset"           # where labeled crops get saved
# # BOARD_SIZE = 800                            # size (px) of the warped top-down board
# # SQUARE_SIZE = BOARD_SIZE // 8                # px per square in the warped image
# # CROP_PADDING = 4                             # extra px around each square crop (helps with tall pieces bleeding over edges, per earlier discussion)
# # MANIFEST_PATH = os.path.join(OUTPUT_DATASET_DIR, "manifest.jsonl")  # optional log of every crop saved, useful for dedup / re-splitting later

# # TAG_OFFSET_MM = 30.5      # <-- measure this on your rig and update it
# # SQUARE_SIZE_MM = 47     # <-- real size of one chess square on your board, measure it
# # BOARD_SIZE_MM = SQUARE_SIZE_MM * 8.0
# # # Tag-to-tag spacing along one edge = true board edge + offset on both ends.
# # TAG_SPACING_MM = BOARD_SIZE_MM + 2 * TAG_OFFSET_MM
# # # Fraction of the tag-to-tag distance to move inward from each tag center
# # # to land exactly on the true board corner. This must be computed relative
# # # to TAG-TO-TAG distance (what shrink_corner's direction vectors actually
# # # measure), not the true board size -- using the wrong denominator here
# # # was a bug caught only by checking the numbers, not by eye.
# # TAG_OFFSET_FRACTION = TAG_OFFSET_MM / TAG_SPACING_MM

# # # ---------------------------------------------------------------------------
# # # Occupancy detection config
# # # ---------------------------------------------------------------------------
# # WARP_SIZE = 800              # matches your existing warp size
# # SQUARE_PX = WARP_SIZE // 8

# # FILES = "abcdefgh"
# # RANKS = "12345678"
# # # ----------------------------- HOMOGRAPHY HOOK -----------------------------

# # def warp_board(raw_frame):
# #     """
# #     PLUG IN YOUR EXISTING HOMOGRAPHY CODE HERE.

# #     Should take a raw camera frame (BGR numpy array) and return a
# #     top-down warped board image of shape (BOARD_SIZE, BOARD_SIZE, 3),
# #     where square (0,0) is consistently the same physical corner every
# #     time (e.g. always a1) -- consistency matters more than which
# #     corner you pick, since labels need to line up with the same
# #     square index across frames.

# #     Placeholder below just resizes the input -- replace this with
# #     your AprilTag corner detection + cv2.getPerspectiveTransform /
# #     cv2.warpPerspective call.
# #     """

# #     ret, frame = cap.read()
# #     if not ret:
# #         return

# #     corners, ids, rejected = detector.detectMarkers(frame)

# #     display = frame.copy()
# #     board_corners = {}

# #     if ids is not None:
# #         cv2.aruco.drawDetectedMarkers(display, corners, ids)

# #         for i, tag_id in enumerate(ids.flatten()):
# #             if tag_id in TAG_ROLES:
# #                 tag_corners = corners[i][0]
# #                 center = tag_corners.mean(axis=0)
# #                 board_corners[TAG_ROLES[tag_id]] = center

# #     if len(board_corners) == 4:
# #         pts = np.array([
# #             board_corners["top-left"],
# #             board_corners["top-right"],
# #             board_corners["bottom-right"],
# #             board_corners["bottom-left"]
# #         ], dtype="float32")

# #         cv2.polylines(display, [pts.astype(int)], isClosed=True, color=(0, 255, 0), thickness=3)

# #         # Shrink inward from tag centers to the actual board corners.
# #         tl, tr, br, bl = pts[0], pts[1], pts[2], pts[3]

# #         def shrink_corner(corner, along_edge_a, along_edge_b, frac):
# #             dir_a = (along_edge_a - corner)
# #             dir_b = (along_edge_b - corner)
# #             return corner + dir_a * frac + dir_b * frac

# #         corrected_tl = shrink_corner(tl, tr, bl, TAG_OFFSET_FRACTION)
# #         corrected_tr = shrink_corner(tr, tl, br, TAG_OFFSET_FRACTION)
# #         corrected_br = shrink_corner(br, bl, tr, TAG_OFFSET_FRACTION)
# #         corrected_bl = shrink_corner(bl, br, tl, TAG_OFFSET_FRACTION)

# #         pts_corrected = np.array([corrected_tl, corrected_tr, corrected_br, corrected_bl],
# #                                   dtype="float32")

# #         cv2.polylines(display, [pts_corrected.astype(int)], isClosed=True,
# #                       color=(255, 0, 255), thickness=2)  # magenta = corrected board outline

# #         size = WARP_SIZE
# #         # warp using raw tag centers, full size, no fudge factor
# #         dst = np.array([[0, 0], [size, 0], [size, size], [0, size]], dtype="float32")
# #         matrix = cv2.getPerspectiveTransform(pts, dst)  # pts = raw tag centers, NOT pts_corrected
# #         warped_full = cv2.warpPerspective(frame, matrix, (size, size))

# #         # THEN crop inward by a fixed pixel margin to go from tag-centers-frame to board-edges-frame
# #         margin = int(size * TAG_OFFSET_FRACTION)  # now safe to apply uniformly
# #         warped = warped_full[margin:size-margin, margin:size-margin]
# #         warped = cv2.resize(warped, (size, size))  # back to your standard board size for slicing
# #         return warped
    
# #     print("Could not find all 4 tags. Make sure the board is fully visible and tags are not occluded.")
# #     return


# # # ----------------------------- LABELING UI -----------------------------

# # class SquareLabeler:
# #     def __init__(self, frame_paths):
# #         self.frame_paths = frame_paths
# #         self.frame_idx = 0
# #         self.occupied = [[False] * 8 for _ in range(8)]  # [row][col], row 0 = top of warped image
# #         self.window_name = "Label squares - click to toggle, 's' save, 'n' skip, 'r' reset, 'q' quit"
# #         self.warped = None
# #         # Window/mouse-callback setup deferred to run() since it requires a
# #         # real display -- keeps this class testable headlessly (e.g.
# #         # exercising save_current()) without needing a GUI.

# #     def _init_window(self):
# #         cv2.namedWindow(self.window_name)
# #         cv2.setMouseCallback(self.window_name, self.on_click)

# #     def on_click(self, event, x, y, flags, param):
# #         if event != cv2.EVENT_LBUTTONDOWN:
# #             return
# #         col = x // SQUARE_SIZE
# #         row = y // SQUARE_SIZE
# #         if 0 <= row < 8 and 0 <= col < 8:
# #             self.occupied[row][col] = not self.occupied[row][col]

# #     def draw_overlay(self):
# #         display = self.warped.copy()
# #         for row in range(8):
# #             for col in range(8):
# #                 x0, y0 = col * SQUARE_SIZE, row * SQUARE_SIZE
# #                 x1, y1 = x0 + SQUARE_SIZE, y0 + SQUARE_SIZE
# #                 cv2.rectangle(display, (x0, y0), (x1, y1), (60, 60, 60), 1)
# #                 if self.occupied[row][col]:
# #                     overlay = display.copy()
# #                     cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 0, 255), -1)
# #                     display = cv2.addWeighted(overlay, 0.35, display, 0.65, 0)
# #         progress = f"Frame {self.frame_idx + 1}/{len(self.frame_paths)}"
# #         cv2.putText(display, progress, (10, BOARD_SIZE - 10),
# #                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
# #         return display

# #     def reset_toggles(self):
# #         self.occupied = [[False] * 8 for _ in range(8)]

# #     def save_current(self):
# #         os.makedirs(os.path.join(OUTPUT_DATASET_DIR, "occupied"), exist_ok=True)
# #         os.makedirs(os.path.join(OUTPUT_DATASET_DIR, "empty"), exist_ok=True)

# #         frame_name = os.path.splitext(os.path.basename(self.frame_paths[self.frame_idx]))[0]
# #         manifest_entries = []

# #         for row in range(8):
# #             for col in range(8):
# #                 x0 = max(col * SQUARE_SIZE - CROP_PADDING, 0)
# #                 y0 = max(row * SQUARE_SIZE - CROP_PADDING, 0)
# #                 x1 = min(x0 + SQUARE_SIZE + 2 * CROP_PADDING, BOARD_SIZE)
# #                 y1 = min(y0 + SQUARE_SIZE + 2 * CROP_PADDING, BOARD_SIZE)
# #                 crop = self.warped[y0:y1, x0:x1]

# #                 label = "occupied" if self.occupied[row][col] else "empty"
# #                 out_name = f"{frame_name}_r{row}c{col}.png"
# #                 out_path = os.path.join(OUTPUT_DATASET_DIR, label, out_name)
# #                 cv2.imwrite(out_path, crop)

# #                 manifest_entries.append({
# #                     "frame": frame_name, "row": row, "col": col,
# #                     "label": label, "path": out_path
# #                 })

# #         with open(MANIFEST_PATH, "a") as f:
# #             for entry in manifest_entries:
# #                 f.write(json.dumps(entry) + "\n")

# #         print(f"[saved] {frame_name}: "
# #               f"{sum(sum(r) for r in self.occupied)} occupied / "
# #               f"{64 - sum(sum(r) for r in self.occupied)} empty")

# #     def run(self):
# #         self._init_window()
# #         while self.frame_idx < len(self.frame_paths):
# #             raw = cv2.imread(self.frame_paths[self.frame_idx])
# #             if raw is None:
# #                 print(f"[skip] couldn't read {self.frame_paths[self.frame_idx]}")
# #                 self.frame_idx += 1
# #                 continue

# #             self.warped = warp_board(raw)
# #             if self.warped is None:
# #                 print(f"[skip] frame {self.frame_idx + 1} - tags not detected")
# #                 self.frame_idx += 1
# #                 continue
# #             self.reset_toggles()

# #             while True:
# #                 display = self.draw_overlay()
# #                 cv2.imshow(self.window_name, display)
# #                 key = cv2.waitKey(20) & 0xFF

# #                 if key == ord('s'):
# #                     self.save_current()
# #                     self.frame_idx += 1
# #                     break
# #                 elif key == ord('n'):
# #                     print(f"[skip, no save] frame {self.frame_idx + 1}")
# #                     self.frame_idx += 1
# #                     break
# #                 elif key == ord('r'):
# #                     self.reset_toggles()
# #                 elif key == ord('q'):
# #                     cv2.destroyAllWindows()
# #                     return

# #         cv2.destroyAllWindows()
# #         print("All frames labeled.")


# # def main():
# #     frame_paths = sorted(glob.glob(os.path.join(INPUT_FRAMES_DIR, "*.jpg")) +
# #                           glob.glob(os.path.join(INPUT_FRAMES_DIR, "*.png")))
# #     if not frame_paths:
# #         print(f"No frames found in {INPUT_FRAMES_DIR}. "
# #               f"Capture some board photos there first (jpg or png).")
# #         return

# #     print(f"Found {len(frame_paths)} frames to label.")
# #     labeler = SquareLabeler(frame_paths)
# #     labeler.run()


# # if __name__ == "__main__":
# #     main()

# """
# label_squares.py

# Interactive tool for labeling chessboard squares as occupied/empty,
# to build a training set for the per-square occupancy CNN.

# WORKFLOW:
#   1. Point this at a folder of captured frames (full chessboard images).
#   2. For each frame: AprilTag-based homography warps it to a top-down
#      view, then this script slices it into a 64-square grid and
#      overlays clickable cells.
#   3. Click a square to toggle occupied (red) / empty (no highlight).
#   4. Press 's' to crop and write out all 64 squares into
#      dataset/occupied/ and dataset/empty/, then auto-advances to the
#      next frame.

# Controls:
#   Left click on a square : toggle occupied/empty
#   's'                    : save labels for this frame + advance
#   'n'                    : skip frame without saving
#   'r'                    : reset all toggles on current frame to empty
#   'q'                    : quit
# """

# import cv2
# import numpy as np
# import os
# import glob
# import json

# aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_16h5)
# detector_params = cv2.aruco.DetectorParameters()
# detector = cv2.aruco.ArucoDetector(aruco_dict, detector_params)

# # NOTE: no cv2.VideoCapture here -- this script only ever processes saved
# # image files from INPUT_FRAMES_DIR. It must never touch a live camera;
# # that was the bug (warp_board was secretly re-reading from a live cap
# # instead of using the loaded frame, so every frame looked "the same
# # but different brightness" -- it WAS a different, live frame each
# # time, just not the one you thought you were labeling).

# # Map tag ID -> role
# TAG_ROLES = {0: "top-left", 1: "top-right", 2: "bottom-right", 3: "bottom-left"}

# # ----------------------------- CONFIG -----------------------------

# INPUT_FRAMES_DIR = "./raw_frames"          # folder of full-board images to label
# OUTPUT_DATASET_DIR = "./dataset"           # where labeled crops get saved
# BOARD_SIZE = 800                            # size (px) of the warped top-down board
# SQUARE_SIZE = BOARD_SIZE // 8                # px per square in the warped image
# CROP_PADDING = 4                             # extra px around each square crop
# MANIFEST_PATH = os.path.join(OUTPUT_DATASET_DIR, "manifest.jsonl")

# TAG_OFFSET_MM = 30.5      # <-- measure this on your rig and update it
# SQUARE_SIZE_MM = 47       # <-- real size of one chess square on your board, measure it
# BOARD_SIZE_MM = SQUARE_SIZE_MM * 8.0
# TAG_SPACING_MM = BOARD_SIZE_MM + 2 * TAG_OFFSET_MM
# TAG_OFFSET_FRACTION = TAG_OFFSET_MM / TAG_SPACING_MM

# WARP_SIZE = 800
# SQUARE_PX = WARP_SIZE // 8

# FILES = "abcdefgh"
# RANKS = "12345678"

# # ----------------------------- HOMOGRAPHY -----------------------------

# def warp_board(raw_frame):
#     """
#     Takes a single still image (already loaded, e.g. via cv2.imread) and
#     returns a top-down warped board image of shape (WARP_SIZE, WARP_SIZE, 3),
#     or None if all 4 AprilTags weren't found.

#     IMPORTANT: this must operate ONLY on raw_frame, the argument passed in.
#     No live camera access here -- this fn needs to be deterministic given
#     a fixed input image, since it's called once per saved file when
#     labeling, and must reproduce the same warp every time for the same file.
#     """
#     frame = raw_frame

#     corners, ids, rejected = detector.detectMarkers(frame)
#     board_corners = {}

#     if ids is not None:
#         for i, tag_id in enumerate(ids.flatten()):
#             if tag_id in TAG_ROLES:
#                 tag_corners = corners[i][0]
#                 center = tag_corners.mean(axis=0)
#                 board_corners[TAG_ROLES[tag_id]] = center

#     if len(board_corners) == 4:
#         pts = np.array([
#             board_corners["top-left"],
#             board_corners["top-right"],
#             board_corners["bottom-right"],
#             board_corners["bottom-left"]
#         ], dtype="float32")

#         size = WARP_SIZE
#         dst = np.array([[0, 0], [size, 0], [size, size], [0, size]], dtype="float32")
#         matrix = cv2.getPerspectiveTransform(pts, dst)
#         warped_full = cv2.warpPerspective(frame, matrix, (size, size))

#         margin = int(size * TAG_OFFSET_FRACTION)
#         warped = warped_full[margin:size - margin, margin:size - margin]
#         warped = cv2.resize(warped, (size, size))
#         return warped

#     print("Could not find all 4 tags. Make sure the board is fully visible and tags are not occluded.")
#     return None


# # ----------------------------- LABELING UI -----------------------------

# class SquareLabeler:
#     def __init__(self, frame_paths):
#         self.frame_paths = frame_paths
#         self.frame_idx = 0
#         self.occupied = [[False] * 8 for _ in range(8)]
#         self.window_name = "Label squares - click to toggle, 's' save, 'n' skip, 'r' reset, 'q' quit"
#         self.warped = None

#     def _init_window(self):
#         cv2.namedWindow(self.window_name)
#         cv2.setMouseCallback(self.window_name, self.on_click)

#     def on_click(self, event, x, y, flags, param):
#         if event != cv2.EVENT_LBUTTONDOWN:
#             return
#         col = x // SQUARE_SIZE
#         row = y // SQUARE_SIZE
#         if 0 <= row < 8 and 0 <= col < 8:
#             self.occupied[row][col] = not self.occupied[row][col]

#     def draw_overlay(self):
#         display = self.warped.copy()
#         for row in range(8):
#             for col in range(8):
#                 x0, y0 = col * SQUARE_SIZE, row * SQUARE_SIZE
#                 x1, y1 = x0 + SQUARE_SIZE, y0 + SQUARE_SIZE
#                 cv2.rectangle(display, (x0, y0), (x1, y1), (60, 60, 60), 1)
#                 if self.occupied[row][col]:
#                     overlay = display.copy()
#                     cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 0, 255), -1)
#                     display = cv2.addWeighted(overlay, 0.35, display, 0.65, 0)
#         progress = f"Frame {self.frame_idx + 1}/{len(self.frame_paths)}"
#         cv2.putText(display, progress, (10, BOARD_SIZE - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
#         return display

#     def reset_toggles(self):
#         self.occupied = [[False] * 8 for _ in range(8)]

#     def save_current(self):
#         os.makedirs(os.path.join(OUTPUT_DATASET_DIR, "occupied"), exist_ok=True)
#         os.makedirs(os.path.join(OUTPUT_DATASET_DIR, "empty"), exist_ok=True)

#         frame_name = os.path.splitext(os.path.basename(self.frame_paths[self.frame_idx]))[0]
#         manifest_entries = []

#         for row in range(8):
#             for col in range(8):
#                 x0 = max(col * SQUARE_SIZE - CROP_PADDING, 0)
#                 y0 = max(row * SQUARE_SIZE - CROP_PADDING, 0)
#                 x1 = min(x0 + SQUARE_SIZE + 2 * CROP_PADDING, BOARD_SIZE)
#                 y1 = min(y0 + SQUARE_SIZE + 2 * CROP_PADDING, BOARD_SIZE)
#                 crop = self.warped[y0:y1, x0:x1]

#                 label = "occupied" if self.occupied[row][col] else "empty"
#                 out_name = f"{frame_name}_r{row}c{col}.png"
#                 out_path = os.path.join(OUTPUT_DATASET_DIR, label, out_name)
#                 cv2.imwrite(out_path, crop)

#                 manifest_entries.append({
#                     "frame": frame_name, "row": row, "col": col,
#                     "label": label, "path": out_path
#                 })

#         with open(MANIFEST_PATH, "a") as f:
#             for entry in manifest_entries:
#                 f.write(json.dumps(entry) + "\n")

#         print(f"[saved] {frame_name}: "
#               f"{sum(sum(r) for r in self.occupied)} occupied / "
#               f"{64 - sum(sum(r) for r in self.occupied)} empty")

#     def run(self):
#         self._init_window()
#         while self.frame_idx < len(self.frame_paths):
#             raw = cv2.imread(self.frame_paths[self.frame_idx])
#             if raw is None:
#                 print(f"[skip] couldn't read {self.frame_paths[self.frame_idx]}")
#                 self.frame_idx += 1
#                 continue

#             self.warped = warp_board(raw)
#             if self.warped is None:
#                 print(f"[skip] frame {self.frame_idx + 1} - tags not detected")
#                 self.frame_idx += 1
#                 continue
#             self.reset_toggles()

#             while True:
#                 display = self.draw_overlay()
#                 cv2.imshow(self.window_name, display)
#                 key = cv2.waitKey(20) & 0xFF

#                 if key == ord('s'):
#                     self.save_current()
#                     self.frame_idx += 1
#                     break
#                 elif key == ord('n'):
#                     print(f"[skip, no save] frame {self.frame_idx + 1}")
#                     self.frame_idx += 1
#                     break
#                 elif key == ord('r'):
#                     self.reset_toggles()
#                 elif key == ord('q'):
#                     cv2.destroyAllWindows()
#                     return

#         cv2.destroyAllWindows()
#         print("All frames labeled.")


# def main():
#     frame_paths = sorted(glob.glob(os.path.join(INPUT_FRAMES_DIR, "*.jpg")) +
#                           glob.glob(os.path.join(INPUT_FRAMES_DIR, "*.png")))
#     if not frame_paths:
#         print(f"No frames found in {INPUT_FRAMES_DIR}. "
#               f"Capture some board photos there first (jpg or png).")
#         return

#     print(f"Found {len(frame_paths)} frames to label.")
#     labeler = SquareLabeler(frame_paths)
#     labeler.run()


# if __name__ == "__main__":
#     main()

"""
label_squares.py

Interactive tool for labeling chessboard squares as occupied/empty,
to build a training set for the per-square occupancy CNN.

WORKFLOW:
  1. Point this at a folder of captured frames (full chessboard images).
  2. For each frame: AprilTag-based homography warps it to a top-down
     view, then this script slices it into a 64-square grid and
     overlays clickable cells.
  3. Click a square to toggle occupied (red) / empty (no highlight).
  4. Press 's' to crop and write out all 64 squares into
     dataset/occupied/ and dataset/empty/, then auto-advances to the
     next frame.

Controls:
  Left click on a square : toggle occupied/empty
  's'                    : save labels for this frame + advance
  'n'                    : skip frame without saving
  'r'                    : reset all toggles on current frame to empty
  'q'                    : quit
"""

import cv2
import numpy as np
import os
import glob
import json
import re

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_16h5)
detector_params = cv2.aruco.DetectorParameters()

detector_params.adaptiveThreshWinSizeMin = 3
detector_params.adaptiveThreshWinSizeMax = 55
detector_params.adaptiveThreshWinSizeStep = 4
detector_params.adaptiveThreshConstant = 7
detector_params.minMarkerPerimeterRate = 0.02
detector_params.maxMarkerPerimeterRate = 4.0
detector_params.polygonalApproxAccuracyRate = 0.05

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
        self.occupied = [[False] * 8 for _ in range(8)]
        self.window_name = "Label squares - click to toggle, 's' save, 'n' skip, 'r' reset, 'q' quit"
        self.warped = None

    def _init_window(self):
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.on_click)

    def on_click(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        col = x // SQUARE_SIZE
        row = y // SQUARE_SIZE
        if 0 <= row < 8 and 0 <= col < 8:
            self.occupied[row][col] = not self.occupied[row][col]

    def draw_overlay(self):
        display = self.warped.copy()
        for row in range(8):
            for col in range(8):
                x0, y0 = col * SQUARE_SIZE, row * SQUARE_SIZE
                x1, y1 = x0 + SQUARE_SIZE, y0 + SQUARE_SIZE
                cv2.rectangle(display, (x0, y0), (x1, y1), (60, 60, 60), 1)
                if self.occupied[row][col]:
                    overlay = display.copy()
                    cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 0, 255), -1)
                    display = cv2.addWeighted(overlay, 0.35, display, 0.65, 0)
        progress = f"Frame {self.frame_idx + 1}/{len(self.frame_paths)}"
        cv2.putText(display, progress, (10, BOARD_SIZE - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        return display

    def reset_toggles(self):
        self.occupied = [[False] * 8 for _ in range(8)]

    def save_current(self):
        os.makedirs(os.path.join(OUTPUT_DATASET_DIR, "occupied"), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DATASET_DIR, "empty"), exist_ok=True)

        frame_name = os.path.splitext(os.path.basename(self.frame_paths[self.frame_idx]))[0]
        manifest_entries = []

        for row in range(8):
            for col in range(8):
                x0 = max(col * SQUARE_SIZE - CROP_PADDING, 0)
                y0 = max(row * SQUARE_SIZE - CROP_PADDING, 0)
                x1 = min(x0 + SQUARE_SIZE + 2 * CROP_PADDING, BOARD_SIZE)
                y1 = min(y0 + SQUARE_SIZE + 2 * CROP_PADDING, BOARD_SIZE)
                crop = self.warped[y0:y1, x0:x1]

                label = "occupied" if self.occupied[row][col] else "empty"
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
              f"{sum(sum(r) for r in self.occupied)} occupied / "
              f"{64 - sum(sum(r) for r in self.occupied)} empty")

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
    '{frame_name}_r{row}c{col}.{ext}' into either dataset/occupied/ or
    dataset/empty/. We only need to check ONE square (r0c0) per frame --
    save_current() always writes all 64 in a single call, so if r0c0
    exists, the whole frame was saved. This avoids needing a separate
    'completed frames' tracking file that could drift out of sync with
    what's actually on disk.
    """
    labeled = set()
    for label_dir in ("occupied", "empty"):
        for path in glob.glob(os.path.join(OUTPUT_DATASET_DIR, label_dir, "*_r0c0.*")):
            base = os.path.basename(path)
            match = re.match(r"(.+)_r0c0\.\w+$", base)
            if match:
                labeled.add(match.group(1))
    return labeled


def main():
    frame_paths = sorted(glob.glob(os.path.join(INPUT_FRAMES_DIR, "*.jpg")) +
                          glob.glob(os.path.join(INPUT_FRAMES_DIR, "*.png")))
    if not frame_paths:
        print(f"No frames found in {INPUT_FRAMES_DIR}. "
              f"Capture some board photos there first (jpg or png).")
        return

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
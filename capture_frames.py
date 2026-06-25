# """
# capture_frames.py

# Live webcam preview -- press SPACE (or 'c') to save the current frame
# to ./raw_frames/, ready to be fed into label_squares.py.

# Use this to build up your dataset: set up some pieces on the board,
# hit capture, rearrange pieces, hit capture again, repeat. Vary
# lighting between batches if you can (lamp on/off, different time of
# day, curtains open/closed, etc.) -- that variety matters more for
# final model robustness than the number of photos.

# Controls:
#   SPACE or 'c' : capture and save current frame
#   'q'          : quit

# Run:
#   python3 capture_frames.py
#   python3 capture_frames.py --camera 1      # if you have multiple webcams and the default picks the wrong one
# """

# import cv2
# import os
# import argparse
# import time

# OUTPUT_DIR = "./raw_frames"


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--camera", type=int, default=0,
#                          help="Camera index (try 0 first; if wrong camera or "
#                               "black screen, try 1, 2, etc.)")
#     args = parser.parse_args()

#     os.makedirs(OUTPUT_DIR, exist_ok=True)

#     cap = cv2.VideoCapture(args.camera)
#     if not cap.isOpened():
#         print(f"Could not open camera index {args.camera}. "
#               f"Try a different --camera index, e.g. --camera 1")
#         return

#     # Count existing frames so repeated runs don't overwrite previous captures
#     existing = [f for f in os.listdir(OUTPUT_DIR) if f.startswith("frame_")]
#     next_idx = len(existing)

#     window_name = "Capture board frames - SPACE to save, 'q' to quit"
#     cv2.namedWindow(window_name)

#     print(f"Camera opened. Saving to {OUTPUT_DIR}/  "
#           f"(starting at frame_{next_idx:04d})")
#     print("Press SPACE or 'c' to capture, 'q' to quit.")

#     last_capture_flash = 0

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("Failed to read from camera.")
#             break

#         display = frame.copy()
#         # Brief white flash + filename overlay for half a second after a
#         # capture, so you get clear visual confirmation it saved.
#         if time.time() - last_capture_flash < 0.4:
#             cv2.rectangle(display, (0, 0),
#                           (display.shape[1], display.shape[0]),
#                           (255, 255, 255), 10)

#         count_text = f"Captured: {next_idx}"
#         cv2.putText(display, count_text, (10, 30),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

#         cv2.imshow(window_name, display)
#         key = cv2.waitKey(1) & 0xFF

#         if key == ord(' ') or key == ord('c'):
#             filename = os.path.join(OUTPUT_DIR, f"frame_{next_idx:04d}.png")
#             cv2.imwrite(filename, frame)
#             print(f"[saved] {filename}")
#             next_idx += 1
#             last_capture_flash = time.time()
#         elif key == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()
#     print(f"Done. {next_idx} total frames in {OUTPUT_DIR}/")


# if __name__ == "__main__":
#     main()

"""
capture_frames.py

Live webcam preview -- press SPACE (or 'c') to save the current frame
to ./raw_frames/, ready to be fed into label_squares.py.

Before saving, this checks that all 4 AprilTags (the same ones used by
warp_board() in label_squares.py) are currently visible. If any are
missing, capture is blocked and the missing tag(s) are shown on screen --
this catches framing/lighting issues live, instead of discovering a bad
photo only after running label_squares.py or test_occupancy_model.py on it.

Use this to build up your dataset: set up some pieces on the board,
hit capture, rearrange pieces, hit capture again, repeat. Vary
lighting between batches if you can (lamp on/off, different time of
day, curtains open/closed, etc.) -- that variety matters more for
final model robustness than the number of photos.

Controls:
  SPACE or 'c' : capture and save current frame (only works when all 4 tags are visible)
  'q'          : quit

Run:
  python3 capture_frames.py
  python3 capture_frames.py --camera 1      # if you have multiple webcams and the default picks the wrong one
"""

import cv2
import os
import argparse
import time

OUTPUT_DIR = "./raw_frames"

# Must match TAG_ROLES in label_squares.py -- this is what gets checked
# live before allowing a capture.
TAG_ROLES = {0: "top-left", 1: "top-right", 2: "bottom-right", 3: "bottom-left"}

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_16h5)
detector_params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, detector_params)


def detect_tag_status(frame):
    """
    Returns (all_found: bool, found_roles: set, missing_roles: set, corners, ids)
    so the caller can both gate capture and draw a helpful overlay.
    """
    corners, ids, rejected = detector.detectMarkers(frame)
    found_roles = set()

    if ids is not None:
        for tag_id in ids.flatten():
            if tag_id in TAG_ROLES:
                found_roles.add(TAG_ROLES[tag_id])

    missing_roles = set(TAG_ROLES.values()) - found_roles
    all_found = len(missing_roles) == 0
    return all_found, found_roles, missing_roles, corners, ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0,
                         help="Camera index (try 0 first; if wrong camera or "
                              "black screen, try 1, 2, etc.)")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Could not open camera index {args.camera}. "
              f"Try a different --camera index, e.g. --camera 1")
        return

    # Count existing frames so repeated runs don't overwrite previous captures
    existing = [f for f in os.listdir(OUTPUT_DIR) if f.startswith("frame_")]
    next_idx = len(existing)

    window_name = "Capture board frames - SPACE to save, 'q' to quit"
    cv2.namedWindow(window_name)

    print(f"Camera opened. Saving to {OUTPUT_DIR}/  "
          f"(starting at frame_{next_idx:04d})")
    print("Press SPACE or 'c' to capture, 'q' to quit.")
    print("Capture is blocked until all 4 AprilTags are visible.")

    last_capture_flash = 0
    last_blocked_flash = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read from camera.")
            break

        all_found, found_roles, missing_roles, corners, ids = detect_tag_status(frame)

        display = frame.copy()

        # Draw detected tag outlines so you can see exactly what the
        # detector sees, same as the live AprilTag preview you already
        # had working elsewhere.
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(display, corners, ids)

        # Brief white flash + filename overlay for half a second after a
        # successful capture, so you get clear visual confirmation it saved.
        if time.time() - last_capture_flash < 0.4:
            cv2.rectangle(display, (0, 0),
                          (display.shape[1], display.shape[0]),
                          (255, 255, 255), 10)

        # Brief red flash if SPACE was pressed but capture was blocked,
        # so a press doesn't just silently do nothing.
        if time.time() - last_blocked_flash < 0.4:
            cv2.rectangle(display, (0, 0),
                          (display.shape[1], display.shape[0]),
                          (0, 0, 255), 10)

        count_text = f"Captured: {next_idx}"
        cv2.putText(display, count_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Main readiness indicator -- green "READY" or red "MISSING: ..."
        # telling you exactly which corner(s) to fix before it'll let you capture.
        if all_found:
            status_text = "READY - all 4 tags visible"
            status_color = (0, 200, 0)
        else:
            status_text = f"NOT READY - missing: {', '.join(sorted(missing_roles))}"
            status_color = (0, 0, 255)
        cv2.putText(display, status_text, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

        cv2.imshow(window_name, display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(' ') or key == ord('c'):
            if all_found:
                filename = os.path.join(OUTPUT_DIR, f"frame_{next_idx:04d}.png")
                cv2.imwrite(filename, frame)
                print(f"[saved] {filename}")
                next_idx += 1
                last_capture_flash = time.time()
            else:
                print(f"[blocked] capture refused -- missing tag(s): {sorted(missing_roles)}. "
                      f"Reposition camera/board so all 4 corners are visible.")
                last_blocked_flash = time.time()
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Done. {next_idx} total frames in {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
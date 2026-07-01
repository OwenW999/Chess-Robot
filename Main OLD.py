"""
main.py

End-to-end runnable demo:
  1. Load camera intrinsics (you must calibrate your specific webcam first
     -- see calibrate_camera.py companion script, or your existing
     calibration if you've already done one for AprilTag pose work).
  2. Capture a frame, detect the 4 corner AprilTags, solve board pose.
  3. Classify all 64 squares as occupied/empty using the 3D prism method.
  4. Show a debug visualization overlay.

Run with: python main.py --camera 0 --calib camera_calib.npz

Press 'q' to quit, 's' to save a snapshot of the current frame +
occupancy overlay (useful for tuning thresholds offline).
"""

import argparse
import sys

import cv2
import numpy as np

from board_pose import BoardPoseEstimator
from occupancy import classify_board, visualize_debug


def load_calibration(path):
    data = np.load(path)
    return data["camera_matrix"], data["dist_coeffs"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--calib", type=str, default="camera_calib.npz",
                         help="npz file with camera_matrix and dist_coeffs")
    parser.add_argument("--edge-threshold", type=float, default=8.0,
                         help="Tune this against your own board/pieces")
    args = parser.parse_args()

    try:
        camera_matrix, dist_coeffs = load_calibration(args.calib)
    except FileNotFoundError:
        print(f"Calibration file '{args.calib}' not found.")
        print("Run calibrate_camera.py first to generate it, or pass --calib"
              " pointing at an existing one.")
        sys.exit(1)

    pose_estimator = BoardPoseEstimator(camera_matrix, dist_coeffs)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Could not open camera index {args.camera}")
        sys.exit(1)

    print("Running. Press 'q' to quit, 's' to save snapshot.")

    last_good_pose = None  # (rvec, tvec) -- reuse briefly if a frame's
                            # tag detection hiccups, to avoid flicker

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Frame grab failed.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        success, rvec, tvec, info = pose_estimator.estimate_pose(gray)

        if success:
            last_good_pose = (rvec, tvec)
            reproj_err = info["mean_reprojection_error_px"]
            status_text = f"Pose OK | reproj err: {reproj_err:.2f}px | tags: {info['used_ids']}"
            status_color = (0, 200, 0) if reproj_err < 2.0 else (0, 165, 255)
        elif last_good_pose is not None:
            rvec, tvec = last_good_pose
            status_text = f"Pose STALE (last good) | {info.get('reason')}"
            status_color = (0, 165, 255)
        else:
            cv2.putText(frame, f"No pose: {info.get('reason')}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.imshow("Chess CV", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        results = classify_board(
            gray, rvec, tvec, camera_matrix, dist_coeffs,
            edge_density_threshold=args.edge_threshold,
        )

        vis = visualize_debug(frame, results)
        cv2.putText(vis, status_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, status_color, 1, cv2.LINE_AA)

        occupied_squares = [sq for sq, r in results.items() if r.occupied]
        print(f"\rOccupied ({len(occupied_squares)}): {sorted(occupied_squares)}",
              end="", flush=True)

        cv2.imshow("Chess CV", vis)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite("snapshot.png", vis)
            print("\nSaved snapshot.png")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
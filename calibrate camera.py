"""
calibrate_camera.py

Standard OpenCV checkerboard calibration to get camera_matrix and
dist_coeffs, which board_pose.py needs to do accurate solvePnP.

Print a checkerboard (e.g. 9x6 internal corners, OpenCV's default sample
pattern works fine: https://github.com/opencv/opencv/blob/4.x/doc/pattern.png),
tape it to something flat, and show it to the webcam from ~15-20 different
angles/distances/tilts, pressing SPACE to capture each. More angle variety
= better calibration. Aim for at least 15-20 good captures.

Run: python calibrate_camera.py --camera 0 --rows 6 --cols 9 --square-mm 25

This matters more than people expect for your use case specifically:
sloppy intrinsics (esp. wrong focal length or unmodeled lens distortion)
directly translates into systematic pose error, which shows up as exactly
the kind of "far side of the board is subtly off" symptom you're fighting
with the depth-of-field blur issue already. Worth doing carefully once.
"""

import argparse

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--rows", type=int, default=6,
                         help="internal corners, not squares, along one side")
    parser.add_argument("--cols", type=int, default=9)
    parser.add_argument("--square-mm", type=float, default=25.0)
    parser.add_argument("--out", type=str, default="camera_calib.npz")
    parser.add_argument("--min-captures", type=int, default=15)
    args = parser.parse_args()

    pattern_size = (args.cols, args.rows)

    objp = np.zeros((args.rows * args.cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:args.cols, 0:args.rows].T.reshape(-1, 2)
    objp *= args.square_mm

    objpoints = []
    imgpoints = []

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("Could not open camera.")
        return

    print(f"Show the checkerboard ({args.cols}x{args.rows} internal corners) "
          f"from varied angles. SPACE = capture, 'q' = finish.")

    img_shape = None
    captures = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img_shape = gray.shape[::-1]

        found, corners = cv2.findChessboardCorners(
            gray, pattern_size,
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        )

        display = frame.copy()
        if found:
            cv2.drawChessboardCorners(display, pattern_size, corners, found)

        cv2.putText(display, f"Captures: {captures}/{args.min_captures}+",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Calibration", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(' ') and found:
            corners_refined = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1),
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            )
            objpoints.append(objp.copy())
            imgpoints.append(corners_refined)
            captures += 1
            print(f"Captured {captures}")
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if captures < args.min_captures:
        print(f"Only {captures} captures -- calibration may be inaccurate. "
              f"Recommend at least {args.min_captures}.")
        if captures < 5:
            print("Too few to calibrate at all. Exiting.")
            return

    print("Running calibration...")
    rms_error, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, img_shape, None, None
    )

    print(f"RMS reprojection error: {rms_error:.4f} px "
          f"(good calibrations are usually < 0.5px)")
    print("Camera matrix:\n", camera_matrix)
    print("Distortion coefficients:\n", dist_coeffs)

    np.savez(args.out, camera_matrix=camera_matrix, dist_coeffs=dist_coeffs,
             rms_error=rms_error)
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
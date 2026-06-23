"""
capture_frames.py

Live webcam preview -- press SPACE (or 'c') to save the current frame
to ./raw_frames/, ready to be fed into label_squares.py.

Use this to build up your dataset: set up some pieces on the board,
hit capture, rearrange pieces, hit capture again, repeat. Vary
lighting between batches if you can (lamp on/off, different time of
day, curtains open/closed, etc.) -- that variety matters more for
final model robustness than the number of photos.

Controls:
  SPACE or 'c' : capture and save current frame
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

    last_capture_flash = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read from camera.")
            break

        display = frame.copy()
        # Brief white flash + filename overlay for half a second after a
        # capture, so you get clear visual confirmation it saved.
        if time.time() - last_capture_flash < 0.4:
            cv2.rectangle(display, (0, 0),
                          (display.shape[1], display.shape[0]),
                          (255, 255, 255), 10)

        count_text = f"Captured: {next_idx}"
        cv2.putText(display, count_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow(window_name, display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(' ') or key == ord('c'):
            filename = os.path.join(OUTPUT_DIR, f"frame_{next_idx:04d}.png")
            cv2.imwrite(filename, frame)
            print(f"[saved] {filename}")
            next_idx += 1
            last_capture_flash = time.time()
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Done. {next_idx} total frames in {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
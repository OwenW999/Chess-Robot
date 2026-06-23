"""
test_occupancy_model.py

Two ways to test your trained model:

1. EVAL MODE (default): runs the model against your full dataset/ folder,
   reports accuracy, and -- more usefully -- shows you the specific
   crops it got WRONG, so you can see what kind of mistakes it's making
   (e.g. consistently missing pawns, confused by shadows, etc.) rather
   than just a single accuracy number.

2. PREDICT MODE: point it at one new full-board image (not yet split into
   squares -- a fresh photo) and it'll run your warp_board() + slicing
   pipeline, classify all 64 squares, and show you a labeled grid so you
   can visually check it against the real board.

Run:
  python3 test_occupancy_model.py                      # eval mode, full dataset/
  python3 test_occupancy_model.py --image path/to/new_frame.jpg   # predict mode
"""

import argparse
import os
import glob

import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as T
import cv2
import numpy as np

from train_occupancy_model import SquareOccupancyCNN, DATASET_DIR, IMAGE_SIZE
from label_squares import warp_board, SQUARE_SIZE, BOARD_SIZE, CROP_PADDING

CHECKPOINT_PATH = "./occupancy_model.pt"


def load_model(device):
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(
            f"No checkpoint found at {CHECKPOINT_PATH}. Run train_occupancy_model.py first."
        )
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    model = SquareOccupancyCNN().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint (trained val_acc: {checkpoint.get('val_acc', '?'):.4f}, "
          f"image_size: {checkpoint.get('image_size', IMAGE_SIZE)})")
    return model


# ----------------------------- EVAL MODE -----------------------------

def eval_on_dataset(model, device):
    """
    Runs the model on every crop in dataset/{occupied,empty}/ and reports
    accuracy plus a breakdown of mistakes. This re-uses the same crops the
    model trained/validated on (it doesn't re-do the train/val split), so
    treat this as a sanity check / error-inspection tool, not a substitute
    for the validation accuracy printed during training -- that number is
    the more honest measure of generalization, since this one can include
    crops the model already saw during training.
    """
    transform = T.Compose([T.Resize((IMAGE_SIZE, IMAGE_SIZE)), T.ToTensor()])

    occupied_paths = sorted(glob.glob(os.path.join(DATASET_DIR, "occupied", "*")))
    empty_paths = sorted(glob.glob(os.path.join(DATASET_DIR, "empty", "*")))
    all_paths = [(p, 1) for p in occupied_paths] + [(p, 0) for p in empty_paths]

    if not all_paths:
        print(f"No crops found in {DATASET_DIR}/. Nothing to evaluate.")
        return

    correct = 0
    mistakes = []  # (path, true_label, predicted_label, confidence)

    with torch.no_grad():
        for path, true_label in all_paths:
            img = Image.open(path).convert("RGB")
            tensor = transform(img).unsqueeze(0).to(device)
            logit = model(tensor)
            prob = torch.sigmoid(logit).item()
            pred_label = 1 if prob > 0.5 else 0

            if pred_label == true_label:
                correct += 1
            else:
                mistakes.append((path, true_label, pred_label, prob))

    total = len(all_paths)
    print(f"\nOverall accuracy on dataset/: {correct}/{total} = {correct/total:.4f}")

    if mistakes:
        print(f"\n{len(mistakes)} mistakes found. Showing up to 10:")
        label_names = {0: "empty", 1: "occupied"}
        for path, true_label, pred_label, prob in mistakes[:10]:
            print(f"  {os.path.basename(path)}: "
                  f"true={label_names[true_label]} predicted={label_names[pred_label]} "
                  f"(confidence={prob:.3f})")
        if len(mistakes) > 10:
            print(f"  ... and {len(mistakes) - 10} more.")
    else:
        print("\nNo mistakes on the full dataset (note: this includes training data, "
              "so a perfect score here doesn't guarantee generalization).")


# ----------------------------- PREDICT MODE -----------------------------

def predict_on_image(model, device, image_path):
    """
    Takes a fresh, full-board photo (NOT pre-sliced), runs it through the
    same warp_board() used during labeling, classifies all 64 squares in
    one batched forward pass, and displays a grid with predictions
    overlaid so you can visually compare against the real board.
    """
    raw = cv2.imread(image_path)
    if raw is None:
        print(f"Could not read image: {image_path}")
        return

    warped = warp_board(raw)
    if warped is None:
        print("Could not detect all 4 AprilTags in this image -- can't warp/predict.")
        return

    transform = T.Compose([T.Resize((IMAGE_SIZE, IMAGE_SIZE)), T.ToTensor()])

    crops = []
    coords = []
    for row in range(8):
        for col in range(8):
            x0 = max(col * SQUARE_SIZE - CROP_PADDING, 0)
            y0 = max(row * SQUARE_SIZE - CROP_PADDING, 0)
            x1 = min(x0 + SQUARE_SIZE + 2 * CROP_PADDING, BOARD_SIZE)
            y1 = min(y0 + SQUARE_SIZE + 2 * CROP_PADDING, BOARD_SIZE)
            crop = warped[y0:y1, x0:x1]
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crop_pil = Image.fromarray(crop_rgb)
            crops.append(transform(crop_pil))
            coords.append((row, col))

    # Batch all 64 crops into a single forward pass -- this is the fast-path
    # benefit of the per-square CNN approach mentioned earlier: one batched
    # call instead of looping 64 times.
    batch = torch.stack(crops).to(device)
    with torch.no_grad():
        logits = model(batch)
        probs = torch.sigmoid(logits).cpu().numpy()

    display = warped.copy()
    occupied_count = 0
    for (row, col), prob in zip(coords, probs):
        x0, y0 = col * SQUARE_SIZE, row * SQUARE_SIZE
        x1, y1 = x0 + SQUARE_SIZE, y0 + SQUARE_SIZE
        is_occupied = prob > 0.5
        if is_occupied:
            occupied_count += 1
            overlay = display.copy()
            cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 0, 255), -1)
            display = cv2.addWeighted(overlay, 0.35, display, 0.65, 0)
        cv2.rectangle(display, (x0, y0), (x1, y1), (60, 60, 60), 1)
        # show confidence so you can see borderline calls, not just the
        # final yes/no -- useful for spotting squares the model is unsure about
        cv2.putText(display, f"{prob:.2f}", (x0 + 5, y0 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    print(f"Predicted {occupied_count}/64 squares as occupied.")

    out_path = "prediction_result.png"
    cv2.imwrite(out_path, display)
    print(f"Saved visualization to {out_path} -- open it to compare against the real board.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default=None,
                         help="Path to a fresh full-board photo to run prediction on. "
                              "If omitted, evaluates against the full dataset/ folder instead.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)

    if args.image:
        predict_on_image(model, device, args.image)
    else:
        eval_on_dataset(model, device)


if __name__ == "__main__":
    main()
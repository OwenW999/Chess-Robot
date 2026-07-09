"""
vision.py

Everything related to turning a camera frame into a length-64
empty/white/black grid:
  - loading the trained per-square CNN
  - grabbing a frame, warping it to a top-down board, slicing into squares
  - classifying all 64 squares in one batched forward pass

Depends on warp_board() and the square-size constants from
label_squares.py, and the model class + checkpoint metadata from
train_occupancy_model.py.
"""

import time

import cv2
import torch
from PIL import Image
import torchvision.transforms as T
import chess

from train_occupancy_model import SquareOccupancyCNN, IMAGE_SIZE, CLASS_NAMES
from label_squares import warp_board, SQUARE_SIZE, BOARD_SIZE, CROP_PADDING

DEFAULT_CHECKPOINT_PATH = "./occupancy_model.pt"

# Class indices -- must match train_occupancy_model.CLASS_NAMES order
# exactly: ["empty", "white", "black"] -> [0, 1, 2]. This is the single
# source of truth for that encoding; inference.py imports these constants
# from here rather than redefining them, so the two can't drift apart.
EMPTY, WHITE, BLACK = 0, 1, 2

_TRANSFORM = T.Compose([T.Resize((IMAGE_SIZE, IMAGE_SIZE)), T.ToTensor()])


def load_model(device, checkpoint_path=DEFAULT_CHECKPOINT_PATH):
    """Load the trained CNN once at startup; reuse the returned model object."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = SquareOccupancyCNN().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Guard against a checkpoint trained with a different class order ever
    # silently swapping what "1" and "2" mean downstream.
    checkpoint_classes = checkpoint.get("class_names", CLASS_NAMES)
    if checkpoint_classes != CLASS_NAMES:
        raise RuntimeError(
            f"Checkpoint class order {checkpoint_classes} does not match "
            f"this module's expected CLASS_NAMES {CLASS_NAMES} -- retrain "
            f"or reconcile the mismatch before continuing."
        )
    return model


def get_board_grid(model, device, cap, timeout=5.0):
    """
    Keeps grabbing frames for up to `timeout` seconds until it gets one
    where all 4 AprilTags are visible, then classifies all 64 squares in
    a single batched forward pass.

    Returns a length-64 list indexed by python-chess square number
    (0=a1, ... 63=h8), where each entry is EMPTY, WHITE, or BLACK.
    This indexing matches inference.board_to_grid(), so the two can be
    compared directly with `==`.

    Returns None if no valid frame was captured within the timeout.
    """
    deadline = time.time() + timeout

    while time.time() < deadline:
        ret, frame = cap.read()
        if not ret:
            print("[get_board_grid] Failed to capture frame.")
            time.sleep(0.1)
            continue

        warped = warp_board(frame)
        if warped is None:
            time.sleep(0.1)
            continue

        crops = []
        square_indices = []
        for row in range(8):
            for col in range(8):
                x0 = max(col * SQUARE_SIZE - CROP_PADDING, 0)
                y0 = max(row * SQUARE_SIZE - CROP_PADDING, 0)
                x1 = min(x0 + SQUARE_SIZE + 2 * CROP_PADDING, BOARD_SIZE)
                y1 = min(y0 + SQUARE_SIZE + 2 * CROP_PADDING, BOARD_SIZE)
                crop = warped[y0:y1, x0:x1]
                crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                crops.append(_TRANSFORM(crop_pil))

                # row/col -> chess square mapping. This depends on which
                # physical corner your rig warps to square (0,0) -- check
                # here first if the board ever looks mirrored/rotated.
                rank = row
                file = 7 - col
                square_indices.append(chess.square(file, rank))

        batch = torch.stack(crops).to(device)
        with torch.no_grad():
            logits = model(batch)
            # argmax over the 3 class logits, NOT sigmoid > 0.5 -- the
            # model predicts empty/white/black, not a binary occupied.
            preds = logits.argmax(dim=1).cpu().numpy()

        grid = [EMPTY] * 64
        for sq_idx, pred in zip(square_indices, preds):
            grid[sq_idx] = int(pred)

        return grid

    print(f"[get_board_grid] Could not detect board after {timeout} seconds -- "
          f"is the board fully visible?")
    return None
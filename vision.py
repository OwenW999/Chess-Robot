import torch
import chess
import numpy as np
import cv2
from PIL import Image
import torchvision.transforms as T

from train_occupancy_model import SquareOccupancyCNN, IMAGE_SIZE
from label_squares import warp_board, SQUARE_SIZE, BOARD_SIZE, CROP_PADDING

CHECKPOINT_PATH = "./occupancy_model.pt"

def load_model(device):
    """Load the trained CNN — call once at startup, reuse the model object."""
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    model = SquareOccupancyCNN().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model

def get_board_grid(model, device, cap, timeout=5.0):
    """Keep trying to capture a valid board grid for up to `timeout` seconds."""
    import time
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

        transform = T.Compose([T.Resize((IMAGE_SIZE, IMAGE_SIZE)), T.ToTensor()])
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
                crops.append(transform(crop_pil))

                rank = row
                file = 7 - col
                square_indices.append(chess.square(file, rank))

        batch = torch.stack(crops).to(device)
        with torch.no_grad():
            logits = model(batch)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()

        grid = [False] * 64
        for sq_idx, prob in zip(square_indices, probs):
            grid[sq_idx] = bool(prob > 0.5)

        return grid  # success

    print("[get_board_grid] Could not detect board after 5 seconds — is the board visible?")
    return None
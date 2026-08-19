"""
chess_move_calc.py

Combines the trained empty/white/black per-square CNN with python-chess +
Stockfish to run a human-vs-engine game:

  1. Wait for the human to make a move, signaled by pressing SPACE.
  2. Scan the board with the camera + CNN into a length-64 grid of
     EMPTY / WHITE / BLACK per square.
  3. Find the legal move whose resulting position matches that grid
     (see infer_move() -- color awareness is what fixes the ambiguous-
     move problem you were hitting with occupancy alone).
  4. Push the human's move, ask Stockfish for a reply, push that too.

Run:
  python chess_move_calc.py
"""

import time

import cv2
import chess
import chess.engine
import torch
from PIL import Image
import torchvision.transforms as T

from train_occupancy_model import SquareOccupancyCNN, IMAGE_SIZE, CLASS_NAMES
from label_squares import warp_board, SQUARE_SIZE, BOARD_SIZE, CROP_PADDING

CHECKPOINT_PATH = "./occupancy_model.pt"
STOCKFISH_PATH = r"StockFish\stockfish"
CAMERA_INDEX = 0

# Class indices -- these MUST line up with train_occupancy_model.CLASS_NAMES,
# which is ["empty", "white", "black"] at indices [0, 1, 2]. load_model()
# below double-checks this against the checkpoint at startup.
EMPTY, WHITE, BLACK = 0, 1, 2

_TRANSFORM = T.Compose([T.Resize((IMAGE_SIZE, IMAGE_SIZE)), T.ToTensor()])


# ----------------------------- VISION -----------------------------

def load_model(device):
    """Load the trained CNN once at startup; reuse the model object across calls."""
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    model = SquareOccupancyCNN().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Guard against a checkpoint trained with a different class order ever
    # silently swapping what "1" and "2" mean here.
    checkpoint_classes = checkpoint.get("class_names", CLASS_NAMES)
    if checkpoint_classes != CLASS_NAMES:
        raise RuntimeError(
            f"Checkpoint class order {checkpoint_classes} does not match "
            f"this script's expected CLASS_NAMES {CLASS_NAMES} -- retrain "
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
    This indexing matches board_to_grid() below, so the two can be
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

                # row/col -> chess square mapping. Kept identical to your
                # original code -- this depends on which physical corner
                # your board is warped to be square (0,0), so don't change
                # it here without re-checking against your camera setup.
                rank = row
                file = 7 - col
                square_indices.append(chess.square(file, rank))

        batch = torch.stack(crops).to(device)
        with torch.no_grad():
            logits = model(batch)
            # argmax over the 3 class logits, NOT sigmoid > 0.5 -- the
            # model now predicts empty/white/black, not a binary occupied
            preds = logits.argmax(dim=1).cpu().numpy()

        grid = [EMPTY] * 64
        for sq_idx, pred in zip(square_indices, preds):
            grid[sq_idx] = int(pred)

        return grid

    print(f"[get_board_grid] Could not detect board after {timeout} seconds -- "
          f"is the board fully visible?")
    return None


def board_to_grid(board):
    """
    Converts a python-chess Board into the SAME length-64, square-indexed,
    EMPTY/WHITE/BLACK representation that get_board_grid() produces from
    the camera, so infer_move() can compare them directly.
    """
    grid = [EMPTY] * 64
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is None:
            grid[square] = EMPTY
        elif piece.color == chess.WHITE:
            grid[square] = WHITE
        else:
            grid[square] = BLACK
    return grid


# ----------------------------- MOVE INFERENCE -----------------------------

def infer_move(board, curr_grid):
    """
    Finds the legal move whose resulting position matches the observed
    grid exactly -- occupancy AND piece color per square.

    This is what color-awareness actually buys you: with occupancy alone,
    two moves that empty/fill the same squares (regardless of whose piece
    ends up where) looked identical. Now a candidate move only counts as
    a match if the mover's color shows up on the destination square and
    the origin square reads back empty -- so most of those old ambiguous
    cases resolve to a single match on their own.

    A few genuine ambiguities can still exist in principle (e.g. two same-
    color, same-piece-type moves that happen to produce an identical final
    board, or underpromotions -- this grid tracks color/occupancy, not
    piece type, so it can't distinguish "promoted to queen" vs "promoted
    to rook" by itself). Those get logged so you can see it happened.
    """
    matches = []
    for move in board.legal_moves:
        test_board = board.copy()
        test_board.push(move)
        if board_to_grid(test_board) == curr_grid:
            matches.append(move)

    if len(matches) == 1:
        return matches[0]
    elif len(matches) == 0:
        return None  # no legal move explains what the camera saw
    else:
        print(f"[infer_move] {len(matches)} legal moves match the observed board -- "
              f"ambiguous. Candidates: {[board.san(m) for m in matches]}")
        return matches[0]  # best-effort: still a legal move consistent with what was seen


# ----------------------------- MAIN LOOP -----------------------------

def wait_for_spacebar(window_name, cap):
    """
    Shows a live camera preview and blocks until SPACE is pressed (or 'q'
    to quit). A visible cv2 window with focus is required for cv2.waitKey
    to actually receive keyboard input on most platforms/backends -- this
    also fixes the original bug where `key` was only read once, before the
    loop, so the loop checked the same stale value forever.
    """
    while True:
        ret, frame = cap.read()
        if ret:
            cv2.imshow(window_name, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            return True
        if key == ord('q'):
            return False


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {CAMERA_INDEX}")

    board = chess.Board()
    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    window_name = "Chess Robot - press SPACE after your move, 'q' to quit"

    try:
        print("Confirming starting position against the camera...")
        start_grid = get_board_grid(model, device, cap)
        if start_grid is not None and start_grid != board_to_grid(board):
            print("WARNING: what the camera sees doesn't match a fresh "
                  "chess.Board() starting position -- double check piece "
                  "placement before playing.")

        while not board.is_game_over():
            print("Your turn -- make your move, then press SPACE (or 'q' to quit).")
            if not wait_for_spacebar(window_name, cap):
                break

            curr_grid = get_board_grid(model, device, cap)
            if curr_grid is None:
                print("Couldn't get a clean board read -- try again.")
                continue

            human_move = infer_move(board, curr_grid)
            if human_move is None:
                print("Couldn't match the observed board to any legal move -- "
                      "check the board and press SPACE to re-scan.")
                continue

            print(f"Detected move: {board.san(human_move)}")
            board.push(human_move)

            if board.is_game_over():
                break

            result = engine.play(board, chess.engine.Limit(time=1.0))
            print(f"Engine plays: {board.san(result.move)}")
            board.push(result.move)
            # TODO: send result.move to your arm controller here

        print(f"Game over: {board.result()}")

    finally:
        engine.quit()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
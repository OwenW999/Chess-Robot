"""
main.py

Runs a full human-vs-Stockfish game using the camera + CNN for move
detection:

  1. Wait for the human to make a move, signaled by pressing SPACE.
  2. Scan the board with vision.get_board_grid() into an empty/white/black
     grid.
  3. Use inference.infer_move() to find the legal move matching that grid.
  4. Push the human's move, ask engine.get_best_move() for a reply, push
     that too.

Run:
  python main.py
"""

import cv2
import chess
import torch

import vision
import engine
from inference import board_to_grid, infer_move
from servo_control import ArmController
from move_executor import MoveExecutor
import chess.engine

CAMERA_INDEX = 0


def wait_for_spacebar(window_name, cap):
    """
    Shows a live camera preview and blocks until SPACE is pressed (or 'q'
    to quit). A visible cv2 window with focus is required for cv2.waitKey
    to actually receive keyboard input on most platforms/backends.
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
    model = vision.load_model(device)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {CAMERA_INDEX}")

    board = chess.Board()
    sf_engine = engine.start_engine()
    window_name = "Chess Robot - press SPACE after your move, 'q' to quit"

    arm = ArmController()
    executor = MoveExecutor(arm)

    try:
        print("Confirming starting position against the camera...")
        arm.camera_clear()
        arm.wait_for_move_completion()
        start_grid = vision.get_board_grid(model, device, cap)
        if start_grid is not None and start_grid != board_to_grid(board):
            print("WARNING: what the camera sees doesn't match a fresh "
                  "chess.Board() starting position -- double check piece "
                  "placement before playing.")

        while not board.is_game_over():
            print("Your turn -- make your move, then press SPACE (or 'q' to quit).")
            if not wait_for_spacebar(window_name, cap):
                break

            curr_grid = vision.get_board_grid(model, device, cap)
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

            engine_move = engine.get_best_move(sf_engine, board)
            print(f"Engine plays: {board.san(engine_move)}")
            executor.execute_move(board, engine_move)  # this pushes the move internally
            arm.camera_clear()
            arm.wait_for_move_completion()

        print(f"Game over: {board.result()}")

    finally:
        engine.stop_engine(sf_engine)
        try:
            arm.stow()
        except Exception as e:
            print(f"Warning: couldn't stow arm on exit: {e}")
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
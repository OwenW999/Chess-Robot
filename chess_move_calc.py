import chess
import chess.engine
import cv2

board = chess.Board()
engine = chess.engine.SimpleEngine.popen_uci(r"StockFish\stockfish")

def infer_move(board, prev_grid, curr_grid):
    """Find the legal move whose effect matches the observed occupancy diff."""
    for move in board.legal_moves:
        test_board = board.copy()
        test_board.push(move)
        predicted_grid = board_to_grid(test_board)  # your function
        if predicted_grid == curr_grid:
            return move
    return None  # no match — ambiguous or vision error, ask for re-scan

# Human's turn
prev_grid = capture_grid()

key = cv2.waitKey(1) & 0xFF

while True:
    if key == ord(' '):
        curr_grid = capture_grid()
        break

human_move = infer_move(board, prev_grid, curr_grid)
board.push(human_move)

# Engine's turn
result = engine.play(board, chess.engine.Limit(time=1.0))
board.push(result.move)
# → send result.move to your arm controller

engine.quit()
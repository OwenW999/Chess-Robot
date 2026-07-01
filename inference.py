import chess

def board_to_grid(board):
    """
    Convert a python-chess Board to a flat list of 64 booleans.
    Index 0 = a1, index 63 = h8 (python-chess square ordering).
    True = occupied, False = empty.
    """
    return [board.piece_at(sq) is not None for sq in chess.SQUARES]


def grids_match(grid_a, grid_b):
    """Check if two occupancy grids are identical."""
    return all(a == b for a, b in zip(grid_a, grid_b))


def infer_move(board, prev_grid, curr_grid):
    """
    Find the legal move whose effect on occupancy matches what the camera saw.
    
    Returns the matching chess.Move, or None if no legal move fits
    (which means a vision error or ambiguous board state — caller should rescan).
    """
    matches = []

    for move in board.legal_moves:
        test_board = board.copy()
        test_board.push(move)
        predicted_grid = board_to_grid(test_board)
        if grids_match(predicted_grid, curr_grid):
            matches.append(move)

    if len(matches) == 1:
        return matches[0]
    elif len(matches) == 0:
        print("[infer_move] No legal move matches the observed board change.")
        return None
    else:
        # Shouldn't happen in a legal game, but just in case
        print(f"[infer_move] Ambiguous: {len(matches)} moves match. Matches: {matches}")
        return None
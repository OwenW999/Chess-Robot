import chess

from vision import EMPTY, WHITE, BLACK


def board_to_grid(board):
    """
    Convert a python-chess Board to a flat list of 64 ints.
    Index 0 = a1, index 63 = h8 (python-chess square ordering).
    EMPTY (0) = no piece, WHITE (1) = white piece, BLACK (2) = black piece.

    This must produce the same encoding as your vision pipeline's
    get_board_grid() (0=empty/1=white/2=black per square), so the two
    can be compared directly.
    """
    grid = [EMPTY] * 64
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece is None:
            grid[sq] = EMPTY
        elif piece.color == chess.WHITE:
            grid[sq] = WHITE
        else:
            grid[sq] = BLACK
    return grid


STATE_NAMES = {EMPTY: "empty", WHITE: "white", BLACK: "black"}


def describe_grid_diff(prev_grid, curr_grid):
    """
    Returns a list of human-readable lines describing every square whose
    state differs between prev_grid and curr_grid, e.g.:
        e2: white -> empty
        e4: empty -> white
    Used for debug output when infer_move() can't explain what it saw.
    """
    lines = []
    for sq in chess.SQUARES:
        if prev_grid[sq] != curr_grid[sq]:
            lines.append(f"  {chess.square_name(sq)}: "
                         f"{STATE_NAMES[prev_grid[sq]]} -> {STATE_NAMES[curr_grid[sq]]}")
    return lines


def grids_match(grid_a, grid_b):
    """Check if two empty/white/black grids are identical."""
    return all(a == b for a, b in zip(grid_a, grid_b))


def infer_move(board, curr_grid):
    """
    Find the legal move whose resulting position matches the observed
    empty/white/black grid exactly -- occupancy AND piece color per square.

    Color is what actually resolves the ambiguity occupancy alone had:
    e.g. two moves that empty/fill the same squares regardless of whose
    piece ends up where used to look identical. Now a candidate only
    counts as a match if the mover's color shows up on the destination
    square and the vacated square reads back empty, so most of those old
    ambiguous cases collapse to a single match on their own.

    A prev_grid is no longer needed here: matching only depends on
    whether a legal move's RESULTING position equals what the camera
    currently sees, not on the diff between two grids.

    Returns the matching chess.Move, or None if no legal move fits
    (a vision error or a state the model couldn't explain -- caller
    should rescan).
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
        prev_grid = board_to_grid(board)
        diff_lines = describe_grid_diff(prev_grid, curr_grid)
        if diff_lines:
            print("  Observed square changes (before -> after):")
            for line in diff_lines:
                print(line)
        else:
            print("  No square changes detected at all -- camera frame may be "
                  "stale, or the board genuinely looks unchanged.")
        return None
    else:
        # With color factored in, this should be much rarer than with
        # occupancy alone -- it now generally only fires for genuine
        # remaining ambiguities, like underpromotion (queen vs. rook vs.
        # bishop vs. knight all look identical to an occupancy+color grid,
        # since it doesn't track piece type).
        print(f"[infer_move] Ambiguous: {len(matches)} moves match. "
              f"Matches: {[board.san(m) for m in matches]}")
        return None
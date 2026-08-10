import chess
from servo_control import ArmController

GRAVEYARD_SLOTS = [(90, 300, 50), (120, 300, 50), (150, 300, 50)]  # measure these on your setup


class MoveExecutor:
    def __init__(self, arm: ArmController):
        self.arm = arm
        self._graveyard_index = 0

    def _next_graveyard_slot(self):
        slot = GRAVEYARD_SLOTS[self._graveyard_index % len(GRAVEYARD_SLOTS)]
        self._graveyard_index += 1
        return slot

    def _remove_piece(self, square_name):
        """Pick up whatever's on this square and discard it to the graveyard."""
        self.arm.move_to_square(square_name, z=150)  # hover above
        self.arm.wait_for_move_completion()
        self.arm.pick(square_name)
        self.arm.wait_for_move_completion()
        x, y, z = self._next_graveyard_slot()
        self.arm.move_to_cartesian(x, y, z + 95, slow=True)
        self.arm.wait_for_move_completion()
        self.arm.move_to_cartesian(x, y, z, slow=True)
        self.arm.wait_for_move_completion()
        self.arm.move_joint('claw', 98)  # open, release
        self.arm.wait_for_move_completion()
        self.arm.move_to_cartesian(x, y, z + 95, slow=True)

    def execute_move(self, board: chess.Board, move: chess.Move):
        """board must be the state BEFORE this move is pushed."""
        from_sq = chess.square_name(move.from_square)
        to_sq = chess.square_name(move.to_square)
        piece = board.piece_at(move.from_square)
        is_pawn = piece is not None and piece.piece_type == chess.PAWN

        is_capture = board.is_capture(move)
        is_en_passant = board.is_en_passant(move)
        is_castling = board.is_castling(move)

        if is_capture:
            if is_en_passant:
                captured_square = chess.square(
                    chess.square_file(move.to_square),
                    chess.square_rank(move.from_square)
                )
            else:
                captured_square = move.to_square
            self._remove_piece(chess.square_name(captured_square))

        if is_castling:
            self.arm.from_to(from_sq, to_sq, pawn=False)
            if board.is_kingside_castling(move):
                rook_from = 'h1' if piece.color == chess.WHITE else 'h8'
                rook_to = 'f1' if piece.color == chess.WHITE else 'f8'
            else:
                rook_from = 'a1' if piece.color == chess.WHITE else 'a8'
                rook_to = 'd1' if piece.color == chess.WHITE else 'd8'
            self.arm.from_to(rook_from, rook_to, pawn=False)
        else:
            self.arm.from_to(from_sq, to_sq, pawn=is_pawn)

        self.arm.stow()
        board.push(move)
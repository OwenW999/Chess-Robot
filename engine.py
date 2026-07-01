import chess.engine

STOCKFISH_PATH = r"StockFish\stockfish\stockfish-windows-x86-64-avx2.exe"

def get_engine_move(board, think_time=1.0):
    """Ask Stockfish for the best move. Returns a chess.Move."""
    with chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH) as engine:
        result = engine.play(board, chess.engine.Limit(time=think_time))
        return result.move
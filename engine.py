"""
engine.py

Thin wrapper around python-chess's Stockfish UCI interface, so main.py 
doesn't need to know anything about chess.engine directly.
"""

import chess
import chess.engine
import os

DEFAULT_STOCKFISH_PATH = "C:\\Users\\eweng\\OneDrive\\Documents\\Owen\\Coding\\StockFish\\stockfish-windows-x86-64-avx2.exe"
DEFAULT_TIME_LIMIT = 1.0  # seconds Stockfish is given to pick a move


def start_engine(path=DEFAULT_STOCKFISH_PATH):
    """
    Launches Stockfish. Returns the engine handle -- keep it around for
    the whole session and call stop_engine() on it when you're done.
    """
    return chess.engine.SimpleEngine.popen_uci(path)


def get_best_move(engine, board, time_limit=DEFAULT_TIME_LIMIT):
    """
    Asks the engine for its move in the current position.
    Returns a chess.Move -- doesn't push it onto `board` itself, since
    the caller (main.py) may want to print/log it first.
    """
    result = engine.play(board, chess.engine.Limit(time=time_limit))
    return result.move


def stop_engine(engine):
    """Cleanly shuts down the engine subprocess. Always call this before exiting."""
    engine.quit()
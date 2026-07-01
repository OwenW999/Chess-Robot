print("Starting Main.py")

import chess
import torch
import cv2
from vision import load_model, get_board_grid
from inference import infer_move, board_to_grid
from engine import get_engine_move
import keyboard

print("Imports done")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
print(f"Device: {device}")

model = load_model(device)
print("Model loaded")

cap = cv2.VideoCapture(0)
print(f"Camera opened: {cap.isOpened()}")

board = chess.Board()
print("Board ready — starting game loop")

while not board.is_game_over():
    prev_grid = get_board_grid(model, device, cap)
    if prev_grid is None:
        print("Couldn't see board — retrying turn.")
        continue

    print("Make your move, then press Space...")
    keyboard.wait('space')
    print("Space pressed")

    curr_grid = get_board_grid(model, device, cap)
    if curr_grid is None:
        print("Couldn't see board after move — retrying turn.")
        continue

    move = infer_move(board, prev_grid, curr_grid)
    if move is None:
        print("Couldn't read move — try again.")
        continue

    board.push(move)
    print(f"Human played: {move}")

    if board.is_game_over():
        break

    engine_move = get_engine_move(board)
    board.push(engine_move)
    print(f"Engine plays: {engine_move}")              

cap.release()
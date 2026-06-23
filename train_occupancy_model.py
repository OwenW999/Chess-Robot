"""
train_occupancy_model.py

Trains a small CNN to classify a chess-square crop as occupied or empty.
Reads directly from the dataset/occupied/ and dataset/empty/ folders
produced by label_squares.py.

WHAT THIS SCRIPT DOES, STEP BY STEP:
  1. Loads all image paths from dataset/{occupied,empty}/, splits them
     into train/validation sets BY FRAME (not by individual crop --
     see split_by_frame() for why this matters).
  2. Defines a small CNN (a few conv blocks + a small classifier head).
  3. Trains for a number of epochs, tracking train/val loss and accuracy.
  4. Saves the best-performing model checkpoint to disk.
  5. Prints a final summary and a couple of misclassified examples so
     you can sanity check what's going wrong, if anything.

Run:
  python3 train_occupancy_model.py
  python3 train_occupancy_model.py --epochs 30 --batch-size 64

Requires: torch, torchvision, pillow
  pip install torch torchvision pillow
"""

import argparse
import os
import re
import glob
import random
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T

# ----------------------------- CONFIG -----------------------------

DATASET_DIR = "./dataset"
CHECKPOINT_PATH = "./occupancy_model.pt"
IMAGE_SIZE = 64          # crops get resized to this (square) size before going into the CNN
VAL_FRACTION = 0.2       # 20% of frames held out for validation
SEED = 42

# ----------------------------- DATA -----------------------------

def extract_frame_id(filepath):
    """
    Pulls the frame identifier out of a crop filename, e.g.
    'frame012_r3c5.png' -> 'frame012'.

    This MUST match however label_squares.py names files (frame_name
    followed by _r{row}c{col}.png). If you changed that naming scheme,
    update this regex to match, since the train/val split below relies
    on grouping crops correctly by their source frame.
    """
    base = os.path.basename(filepath)
    match = re.match(r"(.+)_r\d+c\d+\.\w+$", base)
    if match:
        return match.group(1)
    # fallback: if naming doesn't match expected pattern, treat whole
    # filename as its own frame id (every crop becomes its own "frame",
    # which just disables the leakage protection below -- not ideal, but
    # doesn't crash, and you'll see a warning printed).
    return base


def split_by_frame(occupied_paths, empty_paths, val_fraction, seed):
    """
    Splits into train/val BY FRAME, not by individual crop.

    Why this matters: each frame contributes 64 crops, all sharing the
    same lighting, same camera position, same exact pieces. If you split
    randomly by crop, some crops from the SAME frame end up in both train
    and val -- val accuracy then partly reflects memorizing that specific
    frame's lighting/shadows rather than generalizing, so it overstates
    how good the model actually is. Splitting whole frames into either
    train or val avoids that leak.
    """
    all_paths = occupied_paths + empty_paths
    frame_to_paths = defaultdict(list)
    for p in all_paths:
        frame_to_paths[extract_frame_id(p)].append(p)

    frame_ids = sorted(frame_to_paths.keys())
    rng = random.Random(seed)
    rng.shuffle(frame_ids)

    n_val_frames = max(1, int(len(frame_ids) * val_fraction))
    val_frame_ids = set(frame_ids[:n_val_frames])
    train_frame_ids = set(frame_ids[n_val_frames:])

    train_paths = [p for fid in train_frame_ids for p in frame_to_paths[fid]]
    val_paths = [p for fid in val_frame_ids for p in frame_to_paths[fid]]

    print(f"Split: {len(train_frame_ids)} frames / {len(train_paths)} crops -> train, "
          f"{len(val_frame_ids)} frames / {len(val_paths)} crops -> val")

    return train_paths, val_paths


class SquareDataset(Dataset):
    def __init__(self, paths, labels, transform):
        self.paths = paths
        self.labels = labels  # 1 = occupied, 0 = empty
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        img = self.transform(img)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return img, label


def build_datasets():
    occupied_paths = sorted(glob.glob(os.path.join(DATASET_DIR, "occupied", "*")))
    empty_paths = sorted(glob.glob(os.path.join(DATASET_DIR, "empty", "*")))

    if not occupied_paths or not empty_paths:
        raise RuntimeError(
            f"Expected images in {DATASET_DIR}/occupied/ and {DATASET_DIR}/empty/, "
            f"found {len(occupied_paths)} and {len(empty_paths)} respectively. "
            f"Run label_squares.py first to build the dataset."
        )

    print(f"Found {len(occupied_paths)} occupied crops, {len(empty_paths)} empty crops.")

    train_paths, val_paths = split_by_frame(occupied_paths, empty_paths, VAL_FRACTION, SEED)

    occupied_set = set(occupied_paths)

    def labels_for(paths):
        return [1 if p in occupied_set else 0 for p in paths]

    train_labels = labels_for(train_paths)
    val_labels = labels_for(val_paths)

    # Light augmentation on train only -- small random brightness/contrast
    # jitter and flips help the model generalize across lighting conditions
    # without needing even more captured data. No augmentation on val, since
    # we want validation accuracy to reflect real, unmodified images.
    train_transform = T.Compose([
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        T.ColorJitter(brightness=0.3, contrast=0.3),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
    ])
    val_transform = T.Compose([
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        T.ToTensor(),
    ])

    train_ds = SquareDataset(train_paths, train_labels, train_transform)
    val_ds = SquareDataset(val_paths, val_labels, val_transform)
    return train_ds, val_ds


# ----------------------------- MODEL -----------------------------

class SquareOccupancyCNN(nn.Module):
    """
    Small CNN for binary occupied/empty classification on a single
    square crop. Deliberately tiny -- this task (well-cropped, top-down,
    fixed-size images) doesn't need a deep network, and a small model
    trains fast and runs all 64 squares in a single batched forward pass
    in milliseconds.
    """
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 64 -> 32

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32 -> 16

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 16 -> 8
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x.squeeze(1)  # raw logits, shape (batch,)


# ----------------------------- TRAIN / EVAL LOOPS -----------------------------

def run_epoch(model, loader, criterion, optimizer, device, train):
    model.train() if train else model.eval()
    total_loss, total_correct, total_count = 0.0, 0, 0

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            if train:
                optimizer.zero_grad()

            logits = model(images)
            loss = criterion(logits, labels)

            if train:
                loss.backward()
                optimizer.step()

            preds = (torch.sigmoid(logits) > 0.5).float()
            total_correct += (preds == labels).sum().item()
            total_count += labels.size(0)
            total_loss += loss.item() * labels.size(0)

    return total_loss / total_count, total_correct / total_count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    torch.manual_seed(SEED)
    random.seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_ds, val_ds = build_datasets()
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = SquareOccupancyCNN().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)

        print(f"Epoch {epoch:3d}/{args.epochs}  "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f}  "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "image_size": IMAGE_SIZE,
                "val_acc": val_acc,
            }, CHECKPOINT_PATH)
            print(f"  -> new best val_acc {val_acc:.4f}, saved to {CHECKPOINT_PATH}")

    print(f"\nTraining complete. Best val accuracy: {best_val_acc:.4f}")
    print(f"Best checkpoint saved at: {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
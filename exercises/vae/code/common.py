"""Shared settings, the MNIST split of the statement, and plotting helpers."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # render straight to files, so the scripts also run without a display
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import random_split
from torchvision import datasets, transforms

HERE = Path(__file__).resolve().parent
FIGURES_DIR = HERE.parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)
DATA_DIR = HERE.parents[3] / "data"  # the repository's data/ folder; torchvision downloads MNIST into data/MNIST

SEED = 42        # torch.manual_seed(SEED) before building each model; also the split generator's seed
EPOCHS = 20      # training budget shared by both models (no early stopping: no decision is taken during training)
BATCH_SIZE = 128
LR = 1e-3
BETA = 1.0
VAL_SEED = 1     # generator of the eps used to evaluate the validation set (the same eps every epoch)
TEST_SEED = 2    # generator of the eps used to evaluate the test set
TRAIN_EVAL_SEED = 3  # generator of the eps used to evaluate the training set once training is over
SIDE = 28        # MNIST images are 28 x 28


def load_mnist():
    """Exercise 1A, exactly as the statement: pixels in [0, 1] (ToTensor), 50 000 / 10 000 split of the official
    training images with generator seed 42; the 10 000 official test images are the test set.

    Returns {"train", "val", "test"}: each a pair (X, y), X of shape (n, 784) and float32, y the digit labels."""
    train_full = datasets.MNIST(DATA_DIR, train=True, download=True, transform=transforms.ToTensor())
    test = datasets.MNIST(DATA_DIR, train=False, download=True, transform=transforms.ToTensor())
    train, val = random_split(train_full, [50_000, 10_000], generator=torch.Generator().manual_seed(SEED))

    def flatten(dataset):
        images, labels = zip(*(dataset[i] for i in range(len(dataset))))
        return torch.stack(images).view(len(dataset), -1), torch.tensor(labels)

    return {"train": flatten(train), "val": flatten(val), "test": flatten(test)}


def neighbour_votes(references, labels, queries, k=10, chunk=1000):
    """For each query, how many of its k nearest references (Euclidean distance) carry each digit: (n, 10).
    A measuring tool only: it says how mixed the digits are around a point."""
    votes = []
    for start in range(0, len(queries), chunk):
        nearest = torch.cdist(queries[start:start + chunk], references).topk(k, largest=False).indices
        votes.append(torch.nn.functional.one_hot(labels[nearest], 10).sum(dim=1))
    return torch.cat(votes)


def grey_fraction(X):
    """Fraction of pixels strictly between 0.1 and 0.9: neither background nor full ink."""
    return float(((X > 0.1) & (X < 0.9)).float().mean())


def tile(images, rows, cols):
    """Arrange rows * cols flattened 28 x 28 images (row-major) into one (rows * 28, cols * 28) array."""
    grid = np.asarray(images, dtype=float).reshape(rows, cols, SIDE, SIDE)
    return grid.transpose(0, 2, 1, 3).reshape(rows * SIDE, cols * SIDE)


def show_grid(ax, images, rows, cols, separators=True):
    """Draw a tiled grid of images in grey scale (0 = black, 1 = white), with a thin line between tiles."""
    ax.imshow(tile(images, rows, cols), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    if separators:
        for k in range(1, cols):
            ax.axvline(k * SIDE - 0.5, color="tab:gray", lw=0.6)
        for k in range(1, rows):
            ax.axhline(k * SIDE - 0.5, color="tab:gray", lw=0.6)


def tile_ticks(ax, axis, labels):
    """Put one tick per tile at its centre along "x" or "y", with the given labels."""
    centres = SIDE * np.arange(len(labels)) + (SIDE - 1) / 2
    if axis == "x":
        ax.set_xticks(centres, labels)
    else:
        ax.set_yticks(centres, labels)


def save(fig, name):
    """Save a figure into docs/exercises/vae/figures/ and release it."""
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / name, dpi=150, bbox_inches="tight")  # keeps legends placed outside the axes
    plt.close(fig)

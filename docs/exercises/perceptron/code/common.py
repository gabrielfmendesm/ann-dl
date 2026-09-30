"""Helpers shared by both exercises: data generation and plotting."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # render straight to files, so the scripts also run without a display
import matplotlib.pyplot as plt
import numpy as np

from perceptron import predict

FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

CLASS_COLORS = ("tab:blue", "tab:orange")


def two_gaussians(rng, mean0, mean1, cov, n_per_class=1000):
    """Draw n_per_class 2D points per class from two multivariate normals.

    Returns X and y in generation order (all of Class 0, then all of Class 1)
    and a random presentation order, drawn once from the shared generator,
    that interleaves the two classes. Training visits the samples in that fixed
    order, so every epoch and every run sees them in exactly the same sequence.
    """
    X0 = rng.multivariate_normal(mean0, cov, size=n_per_class)
    X1 = rng.multivariate_normal(mean1, cov, size=n_per_class)
    X = np.vstack([X0, X1])
    y = np.concatenate([np.zeros(n_per_class, dtype=int), np.ones(n_per_class, dtype=int)])
    order = rng.permutation(len(X))
    return X, y, order


def scatter_classes(ax, X, y, size=8, alpha=0.5):
    """Scatter plot with one color per class."""
    for k in (0, 1):
        ax.scatter(*X[y == k].T, s=size, alpha=alpha, color=CLASS_COLORS[k], label=f"Class {k}")


def draw_boundary(ax, w, b, xlim, **style):
    """Draw the decision boundary w1*x1 + w2*x2 + b = 0 across the x-range xlim."""
    x1 = np.array(xlim, dtype=float)
    if abs(w[1]) > 1e-12:
        ax.plot(x1, -(w[0] * x1 + b) / w[1], **style)
    else:  # w2 = 0: the boundary is the vertical line x1 = -b / w1
        ax.axvline(-b / w[0], **style)


def mark_misclassified(ax, X, y, w, b, size=22):
    """Overlay a black 'x' on every point that (w, b) misclassifies; returns their count."""
    wrong = predict(X, w, b) != y
    n_wrong = int(wrong.sum())
    ax.scatter(*X[wrong].T, s=size, marker="x", color="black", linewidths=0.8, label=f"misclassified ({n_wrong})")
    return n_wrong


def data_limits(X, pad=0.8):
    """Axis limits that frame all the points with a margin."""
    return ((X[:, 0].min() - pad, X[:, 0].max() + pad),
            (X[:, 1].min() - pad, X[:, 1].max() + pad))


def save(fig, name):
    """Save a figure into docs/exercises/perceptron/figures/ and release it."""
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / name, dpi=150)
    plt.close(fig)

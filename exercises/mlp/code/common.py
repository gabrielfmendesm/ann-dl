"""Shared settings, the two datasets of the statement, and plotting helpers."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # render straight to files, so the scripts also run without a display
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs, make_classification
from sklearn.model_selection import train_test_split

HERE = Path(__file__).resolve().parent
FIGURES_DIR = HERE.parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

SEED = 42          # every network starts from np.random.default_rng(SEED); datasets and splits use random_state=SEED
EPOCHS = 2000      # training budget shared by Exercises 2, 3 and 4 (one full-batch update per epoch)
LR = 0.5           # learning rate shared by Exercises 2, 3 and 4
CLASS_COLORS = ("tab:blue", "tab:orange", "tab:green")


def blobs_data():
    """Exercise 2: Class 0 is the middle cluster, Class 1 the two clusters on either side of it."""
    centers = np.array([[0., 0.], [3., 3.], [-3., -3.]])
    X, c = make_blobs(n_samples=[500, 250, 250], centers=centers, cluster_std=1.2, random_state=SEED)
    y = (c > 0).astype(int)  # class 0: 1 cluster | class 1: 2 clusters
    return X, y, c


def classification_data(shuffle=True):
    """Exercise 3 and 4: 1500 samples, 4 informative features, 3 classes with 2 clusters each.

    shuffle=False returns the same samples in generation order (cluster by cluster): the
    shuffle, of the rows and of the feature columns, is the generator's last step, so
    everything drawn before it is identical.
    """
    return make_classification(n_samples=1500, n_features=4, n_informative=4, n_redundant=0, n_repeated=0,
                               n_classes=3, n_clusters_per_class=2, class_sep=1.0, random_state=SEED,
                               shuffle=shuffle)


def split(X, y, *extra):
    """80/20 split, stratified by the label, with the fixed seed."""
    return train_test_split(X, y, *extra, test_size=0.2, random_state=SEED, stratify=y)


def best_line_accuracy(X, y):
    """Exact accuracy of the best straight line on (X, y) for a 2D binary problem: a benchmark, not a model.

    Every line classifies by thresholding a projection u . x. Along the sorted projections the best
    threshold comes from cumulative class counts, and the sorted order only changes at the directions
    where two points project to the same value; rotating u once through all of those directions,
    swapping one adjacent pair at each, visits every way a line can split the points.
    """
    n = len(y)
    i, j = np.triu_indices(n, 1)
    d = X[i] - X[j]
    swap = np.arctan2(d[:, 1], d[:, 0]) + np.pi / 2           # u perpendicular to x_i - x_j
    angles = np.concatenate([swap, swap + np.pi]) % (2 * np.pi)
    first, second = np.concatenate([i, i]), np.concatenate([j, j])
    events = np.argsort(angles)
    angles, first, second = angles[events], first[events], second[events]

    theta = (angles[0] + angles[1]) / 2                       # start between the first two events
    order = np.argsort(X @ np.array([np.cos(theta), np.sin(theta)]))
    ys = y[order]
    score = (np.concatenate([[0], np.cumsum(ys == 0)])                    # Class 0 below the threshold
             + np.concatenate([np.cumsum((ys == 1)[::-1])[::-1], [0]]))   # Class 1 above it
    best = int(score.max())
    order, score, labels = order.tolist(), score.tolist(), y.tolist()
    pos = [0] * n
    for k, point in enumerate(order):
        pos[point] = k
    for a, b in zip(np.roll(first, -1).tolist(), np.roll(second, -1).tolist()):
        k = min(pos[a], pos[b])
        assert abs(pos[a] - pos[b]) == 1, "points swapping order must be adjacent"
        low, high = order[k], order[k + 1]
        order[k], order[k + 1] = high, low
        pos[high], pos[low] = k, k + 1
        if labels[low] != labels[high]:                       # only a swap between classes changes a count
            score[k + 1] += 2 if labels[high] == 0 else -2
            best = max(best, score[k + 1])
    return best / n


def scatter_classes(ax, X, y, labels, size=10, alpha=0.6, **kwargs):
    """Scatter plot with one color per class."""
    for k, name in enumerate(labels):
        ax.scatter(*X[y == k].T, s=size, alpha=alpha, color=CLASS_COLORS[k], label=name, **kwargs)


def save(fig, name):
    """Save a figure into docs/exercises/mlp/figures/ and release it."""
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / name, dpi=150, bbox_inches="tight")  # keeps legends placed outside the axes
    plt.close(fig)

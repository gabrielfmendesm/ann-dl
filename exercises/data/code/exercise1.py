"""Exercise 1 — Point Clouds: Geometry and Spread in 2D.

Generates the four Gaussian clouds, studies how the spread (scale factor s)
changes class overlap, and produces Figures 1, 2 and 3 plus the numbers
reported in the text (separation ratios and mixing rates).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# Fixed seed, single generator used throughout (technical rule of the activity)
rng = np.random.default_rng(42)

# Class parameters from the statement: mean and standard deviation per axis
MEANS = np.array([[2.0, 3.0], [5.0, 6.0], [8.0, 1.0], [15.0, 4.0]])
STDS = np.array([[0.8, 2.5], [1.2, 1.9], [0.9, 0.9], [0.5, 2.0]])
N_PER_CLASS = 100
N_CLASSES = 4
SCALES = [0.5, 1.0, 2.0, 4.0]
COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red"]


def generate_dataset(scale):
    """Draw 100 points per class with standard deviations multiplied by `scale`."""
    points, labels = [], []
    for k in range(N_CLASSES):
        points.append(rng.normal(MEANS[k], STDS[k] * scale, size=(N_PER_CLASS, 2)))
        labels.append(np.full(N_PER_CLASS, k))
    return np.vstack(points), np.concatenate(labels)


def scatter_by_class(ax, X, y, marker_size=12):
    """Scatter plot colored by class, with each class mean marked with an X."""
    for k in range(N_CLASSES):
        ax.scatter(*X[y == k].T, s=marker_size, c=COLORS[k], alpha=0.6, label=f"Class {k}")
    for k in range(N_CLASSES):
        ax.scatter(*MEANS[k], marker="X", s=120, c="black", zorder=3)


def mixing_rate(X, y):
    """Fraction of points whose nearest class center is not their own class."""
    dists = np.linalg.norm(X[:, None, :] - MEANS[None, :, :], axis=2)  # (400, 4)
    return float(np.mean(np.argmin(dists, axis=1) != y))


# --- A: generate the four datasets (one per scale; s = 1.0 is the original) ---
datasets = {s: generate_dataset(s) for s in SCALES}

# Figure 1: the original dataset (s = 1.0), centers marked
X1, y1 = datasets[1.0]
fig, ax = plt.subplots(figsize=(8, 6))
scatter_by_class(ax, X1, y1, marker_size=18)
ax.set(title="Figure 1 — Four Gaussian clouds (s = 1.0), class means marked with X",
       xlabel="$x_1$", ylabel="$x_2$")
ax.legend()
fig.tight_layout()
fig.savefig(FIGURES_DIR / "fig1.png", dpi=150)

# --- B: spread study ---
# Figure 2: one subplot per scale, all sharing the same axis limits
fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)
all_points = np.vstack([datasets[s][0] for s in SCALES])
pad = 1.0
axes[0, 0].set_xlim(all_points[:, 0].min() - pad, all_points[:, 0].max() + pad)
axes[0, 0].set_ylim(all_points[:, 1].min() - pad, all_points[:, 1].max() + pad)
for ax, s in zip(axes.flat, SCALES):
    X, y = datasets[s]
    scatter_by_class(ax, X, y, marker_size=8)
    ax.set(title=f"s = {s}", xlabel="$x_1$", ylabel="$x_2$")
axes[0, 0].legend(loc="upper left", fontsize=8)
fig.suptitle("Figure 2 — The same four classes under scale factors s ∈ {0.5, 1, 2, 4} (shared axes)")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "fig2.png", dpi=150)

# Separation ratio r_ij at s = 1: ||mu_i - mu_j|| / (sigma_bar_i + sigma_bar_j)
sigma_bar = STDS.mean(axis=1)  # per-class average of the two axis stds
print("Separation ratios r_ij at s = 1.0:")
r_values = {}
for i in range(N_CLASSES):
    for j in range(i + 1, N_CLASSES):
        r = np.linalg.norm(MEANS[i] - MEANS[j]) / (sigma_bar[i] + sigma_bar[j])
        r_values[(i, j)] = r
        print(f"  r_{i}{j} = ||mu_{i} - mu_{j}|| / (sig_{i} + sig_{j}) "
              f"= {np.linalg.norm(MEANS[i] - MEANS[j]):.3f} / {sigma_bar[i] + sigma_bar[j]:.2f} = {r:.3f}")
smallest_pair = min(r_values, key=r_values.get)
print(f"Smallest: r_{smallest_pair[0]}{smallest_pair[1]} = {r_values[smallest_pair]:.3f}")
print(f"At s = 2 (r scales with 1/s): {r_values[smallest_pair] / 2:.3f}")

# Mixing rate for each scale
rates = {s: mixing_rate(*datasets[s]) for s in SCALES}
print("\nMixing rates:")
for s, rate in rates.items():
    print(f"  s = {s}: {rate:.4f} ({rate:.2%})")

# Figure 3: mixing rate as a function of s
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(SCALES, [rates[s] for s in SCALES], "o-", color="tab:blue", label="mixing rate")
for s in SCALES:
    ax.annotate(f"{rates[s]:.2%}", (s, rates[s]), textcoords="offset points",
                xytext=(8, 6), fontsize=9)
ax.set(title="Figure 3 — Mixing rate as a function of the scale factor s",
       xlabel="scale factor $s$", ylabel="mixing rate")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(FIGURES_DIR / "fig3.png", dpi=150)

# --- C: sketch of plausible decision boundaries over Figure 1 ---
# Nearest-center (Voronoi) regions: a purely geometric proxy for the piecewise
# linear boundaries a network could learn — nothing is trained here.
gx, gy = np.meshgrid(np.linspace(X1[:, 0].min() - 1, X1[:, 0].max() + 1, 500),
                     np.linspace(X1[:, 1].min() - 1, X1[:, 1].max() + 1, 500))
grid = np.stack([gx.ravel(), gy.ravel()], axis=1)
region = np.argmin(np.linalg.norm(grid[:, None, :] - MEANS[None, :, :], axis=2), axis=1)
fig, ax = plt.subplots(figsize=(8, 6))
ax.contourf(gx, gy, region.reshape(gx.shape), levels=[-0.5, 0.5, 1.5, 2.5, 3.5],
            colors=COLORS, alpha=0.12)
ax.contour(gx, gy, region.reshape(gx.shape), levels=[0.5, 1.5, 2.5],
           colors="black", linestyles="dashed", linewidths=1)
scatter_by_class(ax, X1, y1, marker_size=18)
ax.set(title="Figure 1 (annotated) — sketched linear decision boundaries (nearest-center regions)",
       xlabel="$x_1$", ylabel="$x_2$")
ax.legend()
fig.tight_layout()
fig.savefig(FIGURES_DIR / "fig1_boundaries.png", dpi=150)

print("\nFigures saved to", FIGURES_DIR)

"""Exercise 2 — Non-Linearity in Higher Dimensions.

Generates two 5D datasets — shifted Gaussians (Dataset I) and concentric
shells (Dataset II) — projects both with PCA (Figure 4), measures distances
between class centers and radius histograms in 5D (Figure 5), and checks a
simple radial function that separates Dataset II.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# Fixed seed; the generator is consumed in the order the datasets appear below
rng = np.random.default_rng(42)

N_PER_CLASS = 500
DIM = 5

# --- A: Dataset I — shifted Gaussians with the covariances from the statement ---
MU_A = np.zeros(DIM)
SIGMA_A = np.array([
    [1.0, 0.8, 0.1, 0.0, 0.0],
    [0.8, 1.0, 0.3, 0.0, 0.0],
    [0.1, 0.3, 1.0, 0.5, 0.0],
    [0.0, 0.0, 0.5, 1.0, 0.2],
    [0.0, 0.0, 0.0, 0.2, 1.0],
])
MU_B = np.full(DIM, 1.5)
SIGMA_B = np.array([
    [1.5, -0.7, 0.2, 0.0, 0.0],
    [-0.7, 1.5, 0.4, 0.0, 0.0],
    [0.2, 0.4, 1.5, 0.6, 0.0],
    [0.0, 0.0, 0.6, 1.5, 0.3],
    [0.0, 0.0, 0.0, 0.3, 1.5],
])

# Both covariance matrices must be symmetric and positive definite to be valid
for name, cov in [("Sigma_A", SIGMA_A), ("Sigma_B", SIGMA_B)]:
    eig = np.linalg.eigvalsh(cov)
    print(f"{name}: symmetric={np.allclose(cov, cov.T)}, min eigenvalue={eig.min():.3f} (> 0)")

XA = rng.multivariate_normal(MU_A, SIGMA_A, size=N_PER_CLASS)
XB = rng.multivariate_normal(MU_B, SIGMA_B, size=N_PER_CLASS)
X_I = np.vstack([XA, XB])
y_I = np.array([0] * N_PER_CLASS + [1] * N_PER_CLASS)  # 0 = Class A, 1 = Class B


# --- B: Dataset II — concentric shells: random direction on the unit sphere times a radius ---
def shell(radius_mean, radius_std):
    """Points x = rho * u, with u uniform on the unit sphere of R^5 and rho ~ N(mean, std)."""
    v = rng.normal(0.0, 1.0, size=(N_PER_CLASS, DIM))
    u = v / np.linalg.norm(v, axis=1, keepdims=True)  # normalize each direction
    rho = rng.normal(radius_mean, radius_std, size=(N_PER_CLASS, 1))
    return rho * u


XC = shell(2.0, 0.4)  # core
XD = shell(5.0, 0.4)  # shell
X_II = np.vstack([XC, XD])
y_II = np.array([0] * N_PER_CLASS + [1] * N_PER_CLASS)  # 0 = Class C, 1 = Class D

# Sanity checks on the generation
print("\nDataset I sample means:")
print(f"  Class A: {np.round(XA.mean(axis=0), 3)}")
print(f"  Class B: {np.round(XB.mean(axis=0), 3)}")
print(f"  max |sample cov - Sigma| : A = {np.abs(np.cov(XA.T) - SIGMA_A).max():.3f}, "
      f"B = {np.abs(np.cov(XB.T) - SIGMA_B).max():.3f}")
print("Dataset II directions: all unit norm =",
      np.allclose(np.linalg.norm(X_II / np.linalg.norm(X_II, axis=1, keepdims=True), axis=1), 1.0))

# --- C: visualize and compare ---
DATASETS = {
    "Dataset I — shifted Gaussians": (X_I, y_I, ("Class A", "Class B")),
    "Dataset II — concentric shells": (X_II, y_II, ("Class C", "Class D")),
}
COLORS = ("tab:blue", "tab:orange")

# Figure 4: PCA projection of each dataset into 2D, side by side, same axis limits
fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharex=True, sharey=True)
explained = {}
projections = {}
for ax, (name, (X, y, labels)) in zip(axes, DATASETS.items()):
    pca = PCA(n_components=2)
    Z = pca.fit_transform(X)  # PCA fitted on the whole dataset (both classes), unsupervised
    projections[name] = Z
    explained[name] = pca.explained_variance_ratio_
    for k in range(2):
        ax.scatter(*Z[y == k].T, s=10, alpha=0.6, c=COLORS[k], label=labels[k])
    ev = explained[name]
    ax.set(title=f"{name}\nexplained variance: PC1 = {ev[0]:.1%}, PC2 = {ev[1]:.1%}, total = {ev.sum():.1%}",
           xlabel="PC1", ylabel="PC2")
    ax.set_aspect("equal")
    ax.legend()
fig.suptitle("Figure 4 — PCA projection to 2D of both 5D datasets (shared axes)")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "fig4.png", dpi=150)

print("\nExplained variance ratio (PC1, PC2, PC1 + PC2):")
for name, ev in explained.items():
    print(f"  {name}: {ev[0]:.4f}, {ev[1]:.4f}, sum = {ev.sum():.4f}")

# Distance between the class centers, computed in 5D from the samples
print("\nDistance between class centers in 5D (sample means):")
for name, (X, y, labels) in DATASETS.items():
    center_0, center_1 = X[y == 0].mean(axis=0), X[y == 1].mean(axis=0)
    print(f"  {name}: {np.linalg.norm(center_0 - center_1):.4f}")
print(f"  (theoretical: Dataset I = 1.5 * sqrt(5) = {1.5 * np.sqrt(5):.4f}; Dataset II = 0)")

# Figure 5: histogram of the radius ||x|| of every point, both classes overlaid, one panel per dataset
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
print("\nRadius ||x|| per class (mean ± std):")
for ax, (name, (X, y, labels)) in zip(axes, DATASETS.items()):
    radii = np.linalg.norm(X, axis=1)
    bins = np.linspace(0, radii.max() + 0.5, 40)
    for k in range(2):
        ax.hist(radii[y == k], bins=bins, alpha=0.6, color=COLORS[k], label=labels[k])
        print(f"  {name}, {labels[k]}: {radii[y == k].mean():.3f} ± {radii[y == k].std():.3f}")
    ax.set(title=f"{name}", xlabel="radius $\\|x\\|$ (computed in 5D)", ylabel="count")
    ax.legend()
fig.suptitle("Figure 5 — Histogram of the radius of each point, both classes overlaid")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "fig5.png", dpi=150)

# --- D: a simple non-linear function of the inputs that separates Dataset II ---
# f(x) = sum_i x_i^2 - 3.5^2 : negative for the core (radius ~2), positive for the shell (radius ~5)
THRESHOLD = 3.5
f = np.sum(X_II ** 2, axis=1) - THRESHOLD ** 2
predicted = (f > 0).astype(int)
accuracy = np.mean(predicted == y_II)
print(f"\nSeparating function f(x) = sum(x_i^2) - {THRESHOLD}^2 on Dataset II: "
      f"accuracy = {accuracy:.2%} ({int((predicted == y_II).sum())}/{len(y_II)} correct)")

# For contrast: in the 2D PCA plane the classes of Dataset II are nested, not separated
Z_II = projections["Dataset II — concentric shells"]
proj_radius = np.linalg.norm(Z_II, axis=1)
core_max = proj_radius[y_II == 0].max()
inside = np.mean(proj_radius[y_II == 1] < core_max)
print(f"In the PCA plane, {inside:.2%} of Class D points fall inside the disk that contains all of Class C "
      f"(projected radius < {core_max:.3f})")

# And for Dataset I: a nearest-center rule in 5D (purely geometric) already separates most points
centers_I = np.stack([XA.mean(axis=0), XB.mean(axis=0)])
nearest = np.argmin(np.linalg.norm(X_I[:, None, :] - centers_I[None, :, :], axis=2), axis=1)
print(f"Dataset I nearest-center mixing rate in 5D: {np.mean(nearest != y_I):.2%}")

print("\nFigures saved to", FIGURES_DIR)

"""Exercise 2 — a latent space you can see.

Trains the same model with d = 2 (item A), draws the encoded test set and the decoded plane (Figures 4 and 5),
interpolates two test images in pixel space and in the latent space of the d = 16 model (Figure 6), and
measures the numbers the analysis of item E rests on.
"""

import math

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle
from scipy.stats import norm

from common import (BATCH_SIZE, BETA, EPOCHS, LR, SEED, TEST_SEED, VAL_SEED, neighbour_votes, save, show_grid,
                    tile_ticks)
from exercise1 import log_aggregate_posterior
from vae import VAE, evaluate, reconstruction_term, train

D = 2
N_GRID = 15                                 # Figure 5 is a 15 x 15 grid
Q = np.linspace(0.05, 0.95, N_GRID)         # 15 evenly spaced probabilities in [0.05, 0.95]
INTERP = (0, 1)                             # Figure 6 interpolates the first two test images
N_STEPS = 10                                # 10 images per row, t from 0 to 1
K = 10                                      # neighbours in the k-nearest-neighbour vote
DIGIT_COLORS = plt.get_cmap("tab10").colors
DRAW_ORDER_SEED = 0                         # Figure 4 draws the points in a fixed random order


def knn_predict(references, labels, queries, k=K):
    """Majority digit among the k nearest references of each query (ties go to the smaller digit, because
    argmax returns the first maximum). A measuring tool only."""
    return neighbour_votes(references, labels, queries, k).argmax(dim=1)


def ink_area(X):
    """Number of pixels above 0.1 in each image: everything that is not background."""
    return (X > 0.1).sum(dim=1).tolist()


def within_digit_variance(X, y):
    """Spread of the images of the same digit: per-pixel variance across them, summed over the 784 pixels,
    averaged over the ten digits."""
    return float(np.mean([X[y == k].var(dim=0).sum().item() for k in range(10)]))


def run(data, model16, ex1):
    print("\n" + "=" * 72 + "\nEXERCISE 2 — a latent space you can see (d = 2)\n" + "=" * 72)
    X_tr, y_tr = data["train"]
    X_va, _ = data["val"]
    X_te, y_te = data["test"]

    # --- A: the same model and budget, d = 2 -----------------------------------------------------
    torch.manual_seed(SEED)
    model = VAE(D)
    history = train(model, X_tr, X_va, EPOCHS, BATCH_SIZE, LR, BETA, VAL_SEED)
    model.eval()
    test_rec, test_kl = evaluate(model, X_te, TEST_SEED)
    test_rec16, test_kl16 = ex1["test_rec"], ex1["test_kl"]
    print(f"validation after epoch {EPOCHS}: reconstruction {history['val_rec'][-1]:.2f}, "
          f"KL {history['val_kl'][-1]:.2f}")
    print(f"TEST d = {D}: reconstruction {test_rec:.2f}, KL {test_kl:.2f}, negative ELBO {test_rec + test_kl:.2f}")
    print(f"TEST d = 16: reconstruction {test_rec16:.2f}, KL {test_kl16:.2f}, "
          f"negative ELBO {test_rec16 + test_kl16:.2f}")

    # --- B: Figure 4, mu of every test image -----------------------------------------------------
    with torch.no_grad():
        mu_te, _ = model.encode(X_te)
        mu_tr, _ = model.encode(X_tr)
    edge = float(norm.ppf(Q[-1]))  # the grid of Figure 5 spans [-edge, edge] on both axes
    fig, ax = plt.subplots(figsize=(8.2, 7))
    # one scatter in a random order, so that no digit is drawn on top of all the others
    order = torch.randperm(len(X_te), generator=torch.Generator().manual_seed(DRAW_ORDER_SEED))
    ax.scatter(*mu_te[order].T, s=3, alpha=0.5, c=np.array(DIGIT_COLORS)[y_te[order].numpy()], rasterized=True)
    for k in range(10):
        ax.scatter([], [], s=3, color=DIGIT_COLORS[k], label=str(k))  # legend entries
    ax.add_patch(Rectangle((-edge, -edge), 2 * edge, 2 * edge, fill=False, ls="--", lw=1.2, color="black",
                           label=f"grid of Figure 5\n(|z| ≤ {edge:.2f})"))
    ax.set(title=f"Figure 4 — μ of all {len(X_te)} test images (d = 2), colored by digit",
           xlabel="μ₁", ylabel="μ₂", aspect="equal")
    ax.grid(alpha=0.3)
    ax.legend(title="digit", markerscale=4, loc="center left", bbox_to_anchor=(1.02, 0.5))
    save(fig, "fig4.png")

    # --- C: Figure 5, the decoded plane on the inverse-CDF grid ----------------------------------
    z_values = norm.ppf(Q)                                # Phi^-1(q): equal prior mass between neighbours
    Z = torch.tensor([[z1, z2] for z2 in z_values[::-1] for z1 in z_values], dtype=torch.float32)  # top row: z2 max
    with torch.no_grad():
        decoded = model.decode(Z)
    fig, ax = plt.subplots(figsize=(8.6, 9))
    show_grid(ax, decoded, N_GRID, N_GRID, separators=False)
    labels = [f"{z:.2f}" for z in np.round(z_values, 2) + 0.0]  # + 0.0 turns the -0.0 of the centre into 0.0
    tile_ticks(ax, "x", labels)
    tile_ticks(ax, "y", labels[::-1])
    ax.tick_params(axis="x", labelrotation=90)
    ax.set(title=f"Figure 5 — the decoded plane (d = 2): decoder(z) on a {N_GRID} × {N_GRID} grid\n"
                 f"z = Φ⁻¹(q), q = {Q[0]:.2f}, …, {Q[-1]:.2f} on each axis",
           xlabel="z₁ = Φ⁻¹(q₁)", ylabel="z₂ = Φ⁻¹(q₂)")
    save(fig, "fig5.png")
    cell_mass = float((Q[1] - Q[0]) ** 2)
    # for comparison, a linear grid over the same range: prior mass of the tile around its centre and corner
    half = edge / (N_GRID - 1)
    linear_mass = {"centre": float((norm.cdf(half) - norm.cdf(-half)) ** 2),
                   "corner": float((norm.cdf(edge + half) - norm.cdf(edge - half)) ** 2)}
    print(f"grid z values: {np.round(z_values, 3).tolist()}; prior mass of each tile (a slice of width "
          f"{Q[1] - Q[0]:.4f} in q on each axis): {cell_mass:.6f}; on a linear grid over the same range: "
          f"{linear_mass['centre']:.5f} at the centre, {linear_mass['corner']:.5f} at a corner")

    # --- D: Figure 6, interpolation in pixel space and in the latent space of the d = 16 model ---
    a, b = INTERP
    assert y_te[a] != y_te[b], "the two images must show different digits"
    t = torch.linspace(0, 1, N_STEPS)[:, None]
    pixel_row = (1 - t) * X_te[a] + t * X_te[b]
    with torch.no_grad():
        mu16, _ = model16.encode(X_te[[a, b]])
        latent_row = model16.decode((1 - t) * mu16[0] + t * mu16[1])
    fig, ax = plt.subplots(figsize=(10, 2.7))
    show_grid(ax, torch.cat([pixel_row, latent_row]), 2, N_STEPS)
    tile_ticks(ax, "x", [f"{v:.2f}" for v in t.flatten().tolist()])
    tile_ticks(ax, "y", ["pixels", "latent\n(d = 16)"])
    ax.set(title=f"Figure 6 — from test image {a} (a {int(y_te[a])}) to test image {b} (a {int(y_te[b])}): "
                 f"top (1 − t)·x_a + t·x_b, bottom decoder((1 − t)·μ_a + t·μ_b)",
           xlabel="t", ylabel="interpolation")
    save(fig, "fig6.png")

    # --- E: the numbers behind the analysis ------------------------------------------------------
    pred = knn_predict(mu_tr, y_tr, mu_te)
    confusion = np.zeros((10, 10), dtype=int)
    np.add.at(confusion, (y_te.numpy(), pred.numpy()), 1)
    pairs = sorted(((int(confusion[i, j] + confusion[j, i]), i, j) for i in range(10) for j in range(i + 1, 10)),
                   reverse=True)
    count, i, j = pairs[0]
    mixed = ((y_te == i) & (pred == j)) | ((y_te == j) & (pred == i))
    mixed_centre = mu_te[mixed].mean(dim=0).tolist()
    grid_labels = knn_predict(mu_tr, y_tr, Z).view(N_GRID, N_GRID)  # row 0 = top row of Figure 5
    boundary = sorted({(r, c) for r in range(N_GRID) for c in range(N_GRID)
                       for dr, dc in ((0, 1), (1, 0), (0, -1), (-1, 0))
                       if 0 <= r + dr < N_GRID and 0 <= c + dc < N_GRID
                       and {int(grid_labels[r, c]), int(grid_labels[r + dr, c + dc])} == {i, j}})
    with torch.no_grad():
        mu16_tr, logvar16_tr = model16.encode(X_tr)
        mu16_te, _ = model16.encode(X_te)
        path = (1 - t) * mu16[0] + t * mu16[1]  # the codes decoded in the bottom row of Figure 6
        logq_path = log_aggregate_posterior(path, mu16_tr, logvar16_tr).tolist()
        logq_mu16_te = log_aggregate_posterior(mu16_te, mu16_tr, logvar16_tr)  # the same kind of point: a mean
        recon2 = model.decode(mu_te)
        recon16 = model16.decode(mu16_te)
    acc2 = float((pred == y_te).float().mean())
    acc16 = float((knn_predict(mu16_tr, y_tr, mu16_te) == y_te).float().mean())
    spread = {"images": within_digit_variance(X_te, y_te), "d2": within_digit_variance(recon2, y_te),
              "d16": within_digit_variance(recon16, y_te)}
    grid_counts = torch.bincount(grid_labels.flatten(), minlength=10).tolist()
    pixel_digits = knn_predict(X_tr, y_tr, pixel_row).tolist()
    latent_digits = knn_predict(X_tr, y_tr, latent_row).tolist()
    path_percentile = float((logq_mu16_te < min(logq_path)).float().mean())
    ink = {"x_a": ink_area(X_te[[a]])[0], "x_b": ink_area(X_te[[b]])[0],
           "union": int(((X_te[a] > 0.1) | (X_te[b] > 0.1)).sum()),
           "test_mean": float(np.mean(ink_area(X_te))), "pixel_row": ink_area(pixel_row),
           "latent_row": ink_area(latent_row)}

    results = {
        "d": D, "history": history, "test_rec": test_rec, "test_kl": test_kl, "test_neg_elbo": test_rec + test_kl,
        "test_rec_d16": test_rec16, "test_kl_d16": test_kl16,
        "test_rec_from_mu": reconstruction_term(recon2, X_te).item(),
        "kl_bits_d2": test_kl / math.log(2), "digit_bits": math.log2(10),
        "grid_z": z_values.tolist(), "grid_edge": edge, "cell_mass": cell_mass, "linear_grid_mass": linear_mass,
        "interp_indices": [a, b], "interp_digits": [int(y_te[a]), int(y_te[b])], "t": t.flatten().tolist(),
        "knn_k": K, "knn_acc_d2": acc2, "knn_acc_d16": acc16,
        "top_pairs": [{"pair": [p, q], "count": c} for c, p, q in pairs[:6]],
        "overlap_pair": [i, j], "overlap_count": count, "overlap_centre": mixed_centre,
        "grid_labels": grid_labels.tolist(), "grid_label_counts": grid_counts,
        "boundary_cells": [[r + 1, c + 1] for r, c in boundary],  # 1-based (row from the top, column from the left)
        "within_digit_variance": spread,
        "pixel_row_digits": pixel_digits, "latent_row_digits": latent_digits,
        "ink_area": ink, "logq_latent_path": logq_path, "logq_latent_path_min": min(logq_path),
        "logq_test_mu_median": float(logq_mu16_te.median()), "latent_path_min_percentile": path_percentile,
    }
    print(f"\n2E — overlap: {K}-NN on the training codes, test codes misassigned per pair (both directions): "
          + ", ".join(f"{p}-{q}: {c}" for c, p, q in pairs[:6]))
    print(f"     {i}-{j} mixed test codes: {int(mixed.sum())}, centred at mu = ({mixed_centre[0]:.2f}, "
          f"{mixed_centre[1]:.2f})")
    print("     digit of the nearest training codes, for each cell of Figure 5 (top row first):")
    for row in grid_labels.tolist():
        print("       " + " ".join(str(v) for v in row))
    print(f"     cells per digit: {grid_counts}; {i}/{j} boundary cells (row, column), 1-based: "
          f"{results['boundary_cells']}")
    print(f"2E — d = 2 cost: the KL bounds the information in the code: {test_kl:.2f} nats = "
          f"{results['kl_bits_d2']:.1f} bits (the digit alone: log2 10 = {math.log2(10):.1f} bits); "
          f"{K}-NN digit accuracy from mu: d = 2 {acc2:.2%}, d = 16 {acc16:.2%}")
    print(f"     within-digit spread (summed pixel variance): test images {spread['images']:.2f}, reconstructions "
          f"d = 2 {spread['d2']:.2f} ({spread['d2'] / spread['images']:.0%}), d = 16 {spread['d16']:.2f} "
          f"({spread['d16'] / spread['images']:.0%})")
    print(f"2E — Figure 6: digit of the 10 nearest training images, pixel row {pixel_digits}, latent row "
          f"{latent_digits}")
    print(f"     ink area (pixels > 0.1): x_a {ink['x_a']}, x_b {ink['x_b']}, union {ink['union']}, test mean "
          f"{ink['test_mean']:.1f}; pixel row {ink['pixel_row']}; latent row {ink['latent_row']}")
    print(f"     log q(z) along the latent path (d = 16): {[round(v, 2) for v in logq_path]}; lowest "
          f"{min(logq_path):.2f}, above the log q of {path_percentile:.2%} of the test images' mu (median "
          f"{results['logq_test_mu_median']:.2f})")
    return results

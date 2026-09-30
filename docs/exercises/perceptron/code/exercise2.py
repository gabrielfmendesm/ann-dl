"""Exercise 2 — Overlapping data: the case the perceptron cannot solve.

Trains the unchanged Exercise 1 perceptron on two overlapping clouds with the
pocket copy switched on (Figures 4 to 6), and computes the numbers used in the
analysis of item D.
"""

from math import erf, sqrt

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

import perceptron as pc
from common import data_limits, draw_boundary, mark_misclassified, save, scatter_classes, two_gaussians

MEAN0, MEAN1 = [3.0, 3.0], [4.0, 4.0]
COV = [[1.5, 0.0], [0.0, 1.5]]
ETA = 0.01
CENTER = (np.array(MEAN0) + np.array(MEAN1)) / 2  # (3.5, 3.5): the middle of the data cloud


def best_line_accuracy(X, y):
    """Exact accuracy of the best straight line for (X, y): a benchmark, not a model.

    Every line classifies by thresholding a projection u . x (predict 1 when
    u . x >= t). For a fixed direction u the best threshold comes from cumulative
    counts along the sorted projections. The sorted order only changes at the
    directions where two points project to the same value, so rotating u once
    around the circle through all of those directions, swapping one adjacent pair
    at each, visits every possible ordering: the best count seen is the optimum.
    """
    n = len(y)
    i, j = np.triu_indices(n, 1)
    d = X[i] - X[j]
    swap = np.arctan2(d[:, 1], d[:, 0]) + np.pi / 2           # u perpendicular to x_i - x_j
    angles = np.concatenate([swap, swap + np.pi]) % (2 * np.pi)
    first, second = np.concatenate([i, i]), np.concatenate([j, j])
    events = np.argsort(angles)
    angles, first, second = angles[events], first[events], second[events]

    # Start between the first two events, then rotate once around the circle
    theta = (angles[0] + angles[1]) / 2
    order = np.argsort(X @ np.array([np.cos(theta), np.sin(theta)]))
    ys = y[order]
    score = (np.concatenate([[0], np.cumsum(ys == 0)])                   # Class 0 below the threshold
             + np.concatenate([np.cumsum((ys == 1)[::-1])[::-1], [0]]))  # Class 1 above it
    best = int(score.max())
    order, score, labels = order.tolist(), score.tolist(), y.tolist()
    pos = [0] * n
    for k, point in enumerate(order):
        pos[point] = k
    for a, b in zip(np.roll(first, -1).tolist(), np.roll(second, -1).tolist()):
        k = min(pos[a], pos[b])
        assert abs(pos[a] - pos[b]) == 1, "points swapping order must be adjacent"
        low, high = order[k], order[k + 1]          # positions k and k + 1 swap
        order[k], order[k + 1] = high, low
        pos[high], pos[low] = k, k + 1
        if labels[low] != labels[high]:             # only a swap between classes changes a count
            score[k + 1] += 2 if labels[high] == 0 else -2
            best = max(best, score[k + 1])
    return best / n


def bayes_accuracy():
    """Accuracy of the optimal classifier for two Gaussians with equal isotropic covariance.

    The optimal rule is the perpendicular bisector of the means (a straight line),
    and its accuracy is Phi(||mu1 - mu0|| / (2 sigma)).
    """
    z = np.linalg.norm(np.array(MEAN1) - np.array(MEAN0)) / (2 * sqrt(COV[0][0]))
    return 0.5 * (1 + erf(z / sqrt(2)))


def center_offset(w, b):
    """Signed distance from the middle of the cloud to the line (> 0: the middle is classified as Class 1)."""
    return float((w @ CENTER + b) / np.linalg.norm(w))


def describe(name, X, y, w, b):
    """Where a boundary sits relative to the data, and how it classifies each class."""
    pred = pc.predict(X, w, b)
    print(f"  {name}: w = {np.round(w, 6).tolist()}, b = {b:.4f}, accuracy = {np.mean(pred == y):.2%}")
    print(f"      predicted as Class 1: {pred.mean():.2%} of the points | correct on Class 0: "
          f"{np.mean(pred[y == 0] == 0):.2%} | correct on Class 1: {np.mean(pred[y == 1] == 1):.2%} | "
          f"offset of the cloud middle = {center_offset(w, b):+.3f}")


def run(rng):
    print("\n" + "=" * 72 + "\nEXERCISE 2 — overlapping data\n" + "=" * 72)

    # --- A: generate the data ----------------------------------------------------------------
    X_gen, y_gen, order = two_gaussians(rng, MEAN0, MEAN1, COV)
    X, y = X_gen[order], y_gen[order]  # fixed presentation order, classes interleaved
    for k in (0, 1):
        print(f"Class {k}: n = {int(np.sum(y == k))}, sample mean = {X[y == k].mean(axis=0).round(3).tolist()}, "
              f"sample covariance = {np.cov(X[y == k].T).round(3).tolist()}")

    xlim, ylim = data_limits(X)
    fig, ax = plt.subplots(figsize=(7, 6))
    scatter_classes(ax, X, y)
    ax.set(title="Figure 4 — Exercise 2 data\ntwo overlapping Gaussian classes, 1000 points each",
           xlabel="$x_1$", ylabel="$x_2$", xlim=xlim, ylim=ylim)
    ax.legend(loc="upper left")
    save(fig, "fig4.png")

    # --- B: train the unchanged perceptron, with the pocket copy switched on -------------------
    w0, b0 = pc.init_weights(rng)
    print(f"\nInitial weights: w0 = {np.round(w0, 6).tolist()}, b0 = {b0}")
    res = pc.train(X, y, w0, b0, eta=ETA, pocket=True)
    final_w, final_b = res["w"], res["b"]
    pocket = res["pocket"]
    hist = res["history"]
    print(f"Training with eta = {ETA}: {res['epochs']} epochs (converged: {res['converged']}), "
          f"{len(hist['update_positions'])} updates in total")
    describe("final weights ", X, y, final_w, final_b)
    describe(f"pocket weights (best found in epoch {pocket['epoch']})", X, y, pocket["w"], pocket["b"])

    # --- C: figures ----------------------------------------------------------------------------
    # Figure 5: both boundaries over the data; each panel marks the points its own weights get wrong
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.2), sharex=True, sharey=True)
    panels = [(f"Final weights (end of epoch {res['epochs']})", final_w, final_b, "tab:red", pocket["w"], pocket["b"], "tab:green", "pocket"),
              (f"Pocket weights (best, found in epoch {pocket['epoch']})", pocket["w"], pocket["b"], "tab:green",
               final_w, final_b, "tab:red", "final")]
    for ax, (title, w, b, color, w_other, b_other, color_other, other) in zip(axes, panels):
        scatter_classes(ax, X, y, size=6, alpha=0.35)
        n_wrong = mark_misclassified(ax, X, y, w, b, size=12)
        draw_boundary(ax, w, b, xlim, color=color, lw=2.2, label="this boundary")
        draw_boundary(ax, w_other, b_other, xlim, color=color_other, lw=1.3, ls="--", label=f"{other} boundary")
        ax.set(title=f"{title}\naccuracy {pc.accuracy(X, y, w, b):.2%}, {n_wrong} misclassified",
               xlabel="$x_1$", ylabel="$x_2$", xlim=xlim, ylim=ylim)
        ax.legend(loc="upper left", fontsize=9)
    fig.suptitle(f"Figure 5 — Final vs. pocket decision boundaries (η = {ETA})")
    save(fig, "fig5.png")

    # Figure 6: accuracy of the current weights and best-so-far (pocket) accuracy, per epoch
    best_line = best_line_accuracy(X, y)
    epochs = np.arange(len(hist["accuracy"]))
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epochs, hist["accuracy"], color="tab:red", lw=1.2, label="current weights (end of each epoch)")
    ax.plot(epochs, hist["pocket_accuracy"], color="tab:green", lw=2, label="pocket (best so far)")
    ax.plot(pocket["epoch"], pocket["accuracy"], "o", color="tab:green",
            label=f"pocket best: {pocket['accuracy']:.2%} in epoch {pocket['epoch']}")
    ax.axhline(best_line, color="gray", ls="--", lw=1, label=f"best straight line for this sample: {best_line:.2%}")
    ax.axhline(0.5, color="gray", ls=":", lw=1, label="chance: 50%")
    ax.set(title=f"Figure 6 — Accuracy per epoch on overlapping data (η = {ETA})",
           xlabel="epoch", ylabel="accuracy", xlim=(0, epochs[-1]), ylim=(0.45, 0.78))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    save(fig, "fig6.png")

    # --- D: analysis -----------------------------------------------------------------------------
    print("\nItem D:")
    print(f"  best straight line for this sample (exact): {best_line:.2%} ({round(best_line * len(y))} of {len(y)}); "
          f"optimal accuracy for these two distributions: {bayes_accuracy():.2%}")

    # How far b and w move per mistake
    mean_norm = float(np.linalg.norm(X, axis=1).mean())
    print(f"  mean ||x|| = {mean_norm:.3f}: per mistake b moves eta = {ETA}, w moves eta*||x|| = {ETA * mean_norm:.4f} on average")

    # Rebuild (w, b) after every update from the update log: on a mistake the error is +1 for a
    # Class 1 sample and -1 for a Class 0 sample, so the weights are cumulative sums (the model is untouched)
    idx = np.array([i for _, i in hist["update_positions"]])
    err = 2 * y[idx] - 1
    W = w0 + ETA * np.cumsum(err[:, None] * X[idx], axis=0)
    B = b0 + ETA * np.cumsum(err)
    assert np.allclose(W[-1], final_w) and np.isclose(B[-1], final_b)
    norms = np.linalg.norm(W, axis=1)
    print(f"  updates on Class 1 samples (false negatives): {int(np.sum(err > 0))}, "
          f"on Class 0 samples (false positives): {int(np.sum(err < 0))}")
    print(f"  after each update: ||w|| mean {norms.mean():.3f} (5-95%: {np.percentile(norms, 5):.3f}-{np.percentile(norms, 95):.3f}), "
          f"i.e. about {norms.mean() / (ETA * mean_norm):.1f} single updates; b 5-95%: {np.percentile(B, 5):.3f} to {np.percentile(B, 95):.3f}")
    nonzero_b = B[:-1] != 0  # b is exactly 0 whenever the two kinds of mistakes are tied: no relative change there
    print(f"  relative change per mistake (median): ||w|| {np.median(np.abs(np.diff(norms)) / norms[:-1]):.1%}, "
          f"|b| {np.median(np.abs(np.diff(B))[nonzero_b] / np.abs(B[:-1][nonzero_b])):.1%}")
    offsets = (W @ CENTER + B) / norms
    print(f"  shift of the line at the cloud middle per mistake (median): {np.median(np.abs(np.diff(offsets))):.2f}")
    acc_upd = np.concatenate([np.mean(pc.step(X @ W[k:k + 4000].T + B[k:k + 4000]) == y[:, None], axis=0)
                              for k in range(0, len(W), 4000)])
    print(f"  accuracy right after each update: mean {acc_upd.mean():.2%}, 5-95% {np.percentile(acc_upd, 5):.2%}-"
          f"{np.percentile(acc_upd, 95):.2%}, updates leaving it at 70% or more: {np.mean(acc_upd >= 0.70):.2%}, "
          f"highest {acc_upd.max():.2%}")

    # Does the loop settle? Updates, accuracy and weight size at the end of every epoch
    upd = np.array(hist["updates"][1:])
    acc_ep = np.array(hist["accuracy"][1:])
    norm_ep = np.array([np.linalg.norm(w) for w, _ in hist["weights"][1:]])
    print(f"  updates per epoch: min {upd.min()}, mean {upd.mean():.0f}, max {upd.max()}; "
          f"epochs without any update: {int(np.sum(upd == 0))}")
    print(f"  end-of-epoch accuracy over epochs 1-{len(acc_ep)}: min {acc_ep.min():.2%}, mean {acc_ep.mean():.2%}, max {acc_ep.max():.2%}")
    print(f"  end-of-epoch ||w||: first 10 epochs mean {norm_ep[:10].mean():.3f}, last 10 epochs mean "
          f"{norm_ep[-10:].mean():.3f}, max {norm_ep.max():.3f}")

    # What ends each epoch: the sample behind the last update of every epoch
    last_update = {}
    for epoch, i in hist["update_positions"]:
        last_update[epoch] = i  # overwritten until it holds the last update of each epoch
    positions, counts = np.unique(list(last_update.values()), return_counts=True)
    top = int(positions[np.argmax(counts)])
    x_last = X[top]
    print(f"  sample position of the last update of each epoch (position: epochs): "
          f"{dict(zip(positions.tolist(), counts.tolist()))}")
    print(f"      position {top}: Class {y[top]}, x = {x_last.round(3).tolist()}, x1 + x2 = {x_last.sum():.3f}; "
          f"the pocket line predicts Class {pc.predict(x_last[None], pocket['w'], pocket['b'])[0]} for it")
    print(f"  epoch {res['epochs']}, just before its last update: accuracy {pc.accuracy(X, y, W[-2], B[-2]):.2%}, "
          f"offset of the cloud middle {center_offset(W[-2], B[-2]):+.3f}; right after it (final weights): "
          f"accuracy {pc.accuracy(X, y, final_w, final_b):.2%}, offset {center_offset(final_w, final_b):+.3f}")
    pred1 = np.array([pc.predict(X, w, b).mean() for w, b in hist["weights"][1:]])
    offset_ep = np.array([center_offset(w, b) for w, b in hist["weights"][1:]])
    print(f"  end-of-epoch offset of the cloud middle: min {offset_ep.min():+.3f}, max {offset_ep.max():+.3f} "
          f"(positive in {int(np.sum(offset_ep > 0))} of {len(offset_ep)} epochs); share predicted as Class 1: "
          f"{pred1.min():.2%}-{pred1.max():.2%}")

    # Side check: the same points, the same w0 and the same code, presented class by class
    res_sorted = pc.train(X_gen, y_gen, w0, b0, eta=ETA, pocket=True)
    print("\n  Side check — same points presented class-sorted (all of Class 0, then all of Class 1):")
    describe("final weights ", X_gen, y_gen, res_sorted["w"], res_sorted["b"])
    print(f"      pocket: {res_sorted['pocket']['accuracy']:.2%} in epoch {res_sorted['pocket']['epoch']}; "
          f"end-of-epoch accuracy range: {min(res_sorted['history']['accuracy'][1:]):.2%}"
          f"-{max(res_sorted['history']['accuracy'][1:]):.2%}")

    return {"final_w": final_w, "final_b": final_b, "final_accuracy": pc.accuracy(X, y, final_w, final_b),
            "pocket_accuracy": pocket["accuracy"], "pocket_epoch": pocket["epoch"]}

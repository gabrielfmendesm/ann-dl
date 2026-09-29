"""Exercise 1 — Separable data: the case the perceptron was designed for.

Generates the two separable clouds (Figure 1), trains the perceptron with
eta = 0.01 (Figures 2 and 3), re-runs it with eta = 1.0 from the same start,
and checks numerically the zero-start argument of item D.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

import perceptron as pc
from common import data_limits, draw_boundary, mark_misclassified, save, scatter_classes, two_gaussians

MEAN0, MEAN1 = [1.5, 1.5], [5.0, 5.0]
COV = [[0.5, 0.0], [0.0, 0.5]]
ETA, ETA_LARGE = 0.01, 1.0


def unit(w):
    """Direction of the weight vector, w / ||w||."""
    return w / np.linalg.norm(w)


def margin(X, w, b):
    """Distance from the line w . x + b = 0 to the closest sample."""
    return float(np.min(np.abs(X @ w + b)) / np.linalg.norm(w))


def summary(name, X, y, r):
    """One line with everything item D compares between runs."""
    w, b = r["w"], r["b"]
    print(f"  {name}: epochs = {r['epochs']} (converged: {r['converged']}), "
          f"updates per epoch = {r['history']['updates'][1:]}, accuracy = {pc.accuracy(X, y, w, b):.2%}")
    print(f"      w = {np.round(w, 6).tolist()}, b = {b:.4f}, ||w|| = {np.linalg.norm(w):.4f}, "
          f"w/||w|| = {np.round(unit(w), 4).tolist()}, "
          f"line-origin distance = {abs(b) / np.linalg.norm(w):.3f}, margin = {margin(X, w, b):.4f}")


def run(rng):
    print("=" * 72 + "\nEXERCISE 1 — separable data\n" + "=" * 72)

    # --- A: generate the data ----------------------------------------------------------------
    X_gen, y_gen, order = two_gaussians(rng, MEAN0, MEAN1, COV)
    X, y = X_gen[order], y_gen[order]  # fixed presentation order, classes interleaved
    for k in (0, 1):
        print(f"Class {k}: n = {int(np.sum(y == k))}, sample mean = {X[y == k].mean(axis=0).round(3).tolist()}, "
              f"sample covariance = {np.cov(X[y == k].T).round(3).tolist()}")
    # A line that separates every point proves the sample is linearly separable
    print(f"Points on the wrong side of the line x1 + x2 = 6.5 (halfway between the means): "
          f"{int(np.sum((X.sum(axis=1) >= 6.5).astype(int) != y))}")

    xlim, ylim = data_limits(X)
    fig, ax = plt.subplots(figsize=(7, 6))
    scatter_classes(ax, X, y)
    ax.set(title="Figure 1 — Exercise 1 data\ntwo Gaussian classes, 1000 points each",
           xlabel="$x_1$", ylabel="$x_2$", xlim=xlim, ylim=ylim)
    ax.legend(loc="upper left")
    save(fig, "fig1.png")

    # --- B: initial weights, drawn once and shared by every run of this exercise --------------
    w0, b0 = pc.init_weights(rng)
    print(f"\nInitial weights: w0 = {np.round(w0, 6).tolist()}, ||w0|| = {np.linalg.norm(w0):.4f}, b0 = {b0}")

    # --- C: train with eta = 0.01 --------------------------------------------------------------
    res = pc.train(X, y, w0, b0, eta=ETA)
    acc = res["history"]["accuracy"]
    print(f"\nTraining with eta = {ETA}:")
    summary(f"eta = {ETA}", X, y, res)
    print(f"  accuracy after each epoch (epoch 0 = initial weights): {[f'{a:.2%}' for a in acc]}")

    # Figure 2: decision boundary over the data, misclassified points marked
    fig, ax = plt.subplots(figsize=(7, 6))
    scatter_classes(ax, X, y)
    draw_boundary(ax, res["w"], res["b"], xlim, color="black", lw=1.5, label="boundary $w \\cdot x + b = 0$")
    n_wrong = mark_misclassified(ax, X, y, res["w"], res["b"])
    ax.set(title=f"Figure 2 — Decision boundary after training (η = {ETA})\n"
                 f"accuracy {acc[-1]:.2%}, {n_wrong} misclassified points",
           xlabel="$x_1$", ylabel="$x_2$", xlim=xlim, ylim=ylim)
    ax.legend(loc="upper left")
    save(fig, "fig2.png")

    # Figure 3: accuracy x epoch, each point annotated with the updates made during that epoch
    epochs = np.arange(len(acc))
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(epochs, acc, "o-", color="tab:green", label="accuracy on the full dataset")
    for e in epochs:
        if e == 0:
            text, offset, ha, va = f"{acc[e]:.2%}\n(initial weights)", (12, 0), "left", "center"
        else:
            text, offset, ha, va = f"{acc[e]:.2%}\n{res['history']['updates'][e]} updates", (8, -10), "left", "top"
        ax.annotate(text, (e, acc[e]), textcoords="offset points", xytext=offset, ha=ha, va=va, fontsize=9)
    ax.set(title=f"Figure 3 — Accuracy per epoch (η = {ETA})", xlabel="epoch", ylabel="accuracy",
           xticks=epochs, xlim=(-0.4, len(acc) - 0.2), ylim=(0.4, 1.05))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    save(fig, "fig3.png")

    # --- D: analysis -----------------------------------------------------------------------------
    print("\nItem D:")
    # Where the updates of epoch 1 happened inside the epoch (position of the sample)
    pos = np.array([i for e, i in res["history"]["update_positions"] if e == 1])
    print(f"  eta = {ETA}, epoch 1: {len(pos)} updates at sample positions {pos.tolist()}")
    print(f"      first = {pos.min()}, median = {int(np.median(pos))}, last = {pos.max()} (of {len(X)}); "
          f"within the first 100 samples: {int(np.sum(pos < 100))}, first 500: {int(np.sum(pos < 500))}")

    # Re-run with eta = 1.0 and nothing else changed: same data, same order, same w0 and b0
    res_large = pc.train(X, y, w0, b0, eta=ETA_LARGE)
    print(f"\n  Re-run with eta = {ETA_LARGE} (same data, order, w0 and b0):")
    summary(f"eta = {ETA}", X, y, res)
    summary(f"eta = {ETA_LARGE}", X, y, res_large)
    angle = np.degrees(np.arccos(np.clip(unit(res["w"]) @ unit(res_large["w"]), -1.0, 1.0)))
    print(f"  angle between the two boundary directions = {angle:.2f} degrees")

    # How big the start is compared with one update, for each learning rate
    mean_norm = float(np.linalg.norm(X, axis=1).mean())
    print(f"  ||w0|| = {np.linalg.norm(w0):.4f}; mean ||x|| = {mean_norm:.3f}; one update adds eta*||x|| = "
          f"{ETA * mean_norm:.4f} (eta = {ETA}) or {ETA_LARGE * mean_norm:.3f} (eta = {ETA_LARGE})")
    print(f"  effective start w0 / eta: ||w0/{ETA}|| = {np.linalg.norm(w0 / ETA):.3f}, "
          f"||w0/{ETA_LARGE}|| = {np.linalg.norm(w0 / ETA_LARGE):.4f}")

    # Rescaling check: eta = 1 started from w0/0.01 must reproduce the eta = 0.01 run exactly
    res_equiv = pc.train(X, y, w0 / ETA, b0 / ETA, eta=1.0)
    same = res_equiv["history"]["update_positions"] == res["history"]["update_positions"]
    print(f"  eta = 1 from w0/{ETA} vs eta = {ETA} from w0: identical updates = {same}, "
          f"weight ratio = {np.round(res_equiv['w'] / res['w'], 10).tolist()}, "
          f"bias ratio = {res_equiv['b'] / res['b']:.10f}")

    # Zero start: two learning rates, same everything else
    z_small = pc.train(X, y, np.zeros(2), 0.0, eta=ETA)
    z_large = pc.train(X, y, np.zeros(2), 0.0, eta=ETA_LARGE)
    same_zero = z_small["history"]["update_positions"] == z_large["history"]["update_positions"]
    print(f"\n  Zero start (w = 0, b = 0): eta = {ETA} -> epochs {z_small['epochs']}, updates "
          f"{z_small['history']['updates'][1:]}, w = {np.round(z_small['w'], 6).tolist()}, b = {z_small['b']:.4f}")
    print(f"                             eta = {ETA_LARGE}  -> epochs {z_large['epochs']}, updates "
          f"{z_large['history']['updates'][1:]}, w = {np.round(z_large['w'], 6).tolist()}, b = {z_large['b']:.4f}")
    print(f"  identical update sequence = {same_zero}; w ratio = {np.round(z_large['w'] / z_small['w'], 10).tolist()}; "
          f"b ratio = {z_large['b'] / z_small['b']:.10f}; "
          f"same direction = {np.allclose(unit(z_small['w']), unit(z_large['w']))}")
    # The eta = 1.0 run from w0 behaves like the zero start: same updates, weights shifted by w0 only
    same_as_zero = res_large["history"]["update_positions"] == z_large["history"]["update_positions"]
    print(f"  eta = {ETA_LARGE} from w0 vs eta = {ETA_LARGE} from zero: identical updates = {same_as_zero}; "
          f"max |w - w_zero - w0| = {np.abs(res_large['w'] - z_large['w'] - w0).max():.1e}")

    return {"w": res["w"], "b": res["b"], "epochs": res["epochs"], "accuracy": acc[-1],
            "epochs_large": res_large["epochs"],
            "accuracy_large": pc.accuracy(X, y, res_large["w"], res_large["b"])}

"""Exercise 2 — binary classification that a straight line cannot do.

Generates the three-cluster data (Figure 1), fits the logistic-regression
baseline, trains the from-scratch MLP (Figures 2 and 3), and computes the
numbers used in the analysis of item D.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from sklearn.linear_model import LogisticRegression

from common import EPOCHS, LR, SEED, best_line_accuracy, blobs_data, save, scatter_classes, split
from mlp import MLP, BinaryCrossEntropy

LAYERS = [2, 16, 1]
LABELS = ("Class 0", "Class 1")
CLUSTERS = ("middle cluster (Class 0)", "upper-right cluster (Class 1)", "lower-left cluster (Class 1)")


def switches_along(net, a, b, n=2001):
    """Predictions of net on the segment from a to b: the points where the predicted class changes."""
    t = np.linspace(0.0, 1.0, n)
    points = np.outer(1 - t, a) + np.outer(t, b)
    pred = net.predict(points)
    change = np.flatnonzero(np.diff(pred) != 0)
    return pred[0], [((points[k] + points[k + 1]) / 2).round(2).tolist() for k in change]


def per_cluster(pred, y, c):
    return [float(np.mean(pred[c == k] == y[c == k])) for k in range(3)]


def run():
    print("\n" + "=" * 72 + "\nEXERCISE 2 — binary classification\n" + "=" * 72)

    # --- A: data and split ---------------------------------------------------------------------
    X, y, c = blobs_data()
    X_tr, X_te, y_tr, y_te, c_tr, c_te = split(X, y, c)
    print(f"{len(X)} points: Class 0 = {int(np.sum(y == 0))}, Class 1 = {int(np.sum(y == 1))} | "
          f"train {len(X_tr)} (Class 1 share {y_tr.mean():.2%}), test {len(X_te)} (Class 1 share {y_te.mean():.2%})")

    fig, ax = plt.subplots(figsize=(7, 6))
    scatter_classes(ax, X, y, LABELS)
    ax.set(title="Figure 1 — Exercise 2 data: 1000 points\nClass 1 = two clusters on opposite sides of Class 0",
           xlabel="$x_1$", ylabel="$x_2$")
    ax.set_aspect("equal")
    ax.legend(loc="upper left")
    save(fig, "fig1.png")

    # --- B: linear baseline ----------------------------------------------------------------------
    baseline = LogisticRegression().fit(X_tr, y_tr)
    base_pred = baseline.predict(X_te)
    base_acc = float(np.mean(base_pred == y_te))
    w, b = baseline.coef_[0], baseline.intercept_[0]
    p_te = baseline.predict_proba(X_te)[:, 1]
    print(f"\nLogistic-regression baseline: test accuracy {base_acc:.2%}")
    print(f"  w = {np.round(w, 4).tolist()}, b = {b:.4f}, ||w|| = {np.linalg.norm(w):.4f}; "
          f"test probabilities of Class 1 between {p_te.min():.3f} and {p_te.max():.3f}; "
          f"predicted as Class 1: {base_pred.mean():.2%}")
    print("  test accuracy per cluster: "
          + ", ".join(f"{n} {a:.2%}" for n, a in zip(CLUSTERS, per_cluster(base_pred, y_te, c_te))))
    best_tr, best_te = best_line_accuracy(X_tr, y_tr), best_line_accuracy(X_te, y_te)
    print(f"  best straight line (exact): {best_tr:.2%} on the training set, {best_te:.2%} on the test set")

    # --- C: the MLP ------------------------------------------------------------------------------
    net = MLP(LAYERS, BinaryCrossEntropy(), np.random.default_rng(SEED))
    history = net.fit(X_tr, y_tr, EPOCHS, LR)
    pred = net.predict(X_te)
    acc = float(np.mean(pred == y_te))
    train_acc = float(np.mean(net.predict(X_tr) == y_tr))
    print(f"\nMLP {LAYERS}, tanh hidden layer, sigmoid output, binary cross-entropy, full-batch gradient descent, "
          f"lr {LR}, {EPOCHS} epochs, {net.n_parameters()} parameters")
    print(f"  training loss: {history[0]:.4f} at epoch 0, {history[100]:.4f} at epoch 100, "
          f"{history[500]:.4f} at epoch 500, "
          f"{history[-1]:.4f} at epoch {EPOCHS}")
    print(f"  accuracy: train {train_acc:.2%}, test {acc:.2%} "
          f"({int(np.sum(pred != y_te))} of {len(y_te)} test points wrong)")
    print("  test accuracy per cluster: "
          + ", ".join(f"{n} {a:.2%}" for n, a in zip(CLUSTERS, per_cluster(pred, y_te, c_te))))

    # Figure 2: training loss per epoch
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(np.arange(len(history)), history, color="tab:purple", label=f"training loss (MLP {LAYERS})")
    ax.set(title=f"Figure 2 — Exercise 2: training loss per epoch (lr {LR}, full batch)",
           xlabel="epoch", ylabel="binary cross-entropy (training set)", xlim=(0, EPOCHS))
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right")
    save(fig, "fig2.png")

    # Figure 3: decision regions on a grid, test points on top, baseline line for reference
    pad = 1.0
    x1 = np.linspace(X_te[:, 0].min() - pad, X_te[:, 0].max() + pad, 400)
    x2 = np.linspace(X_te[:, 1].min() - pad, X_te[:, 1].max() + pad, 400)
    G1, G2 = np.meshgrid(x1, x2)
    region = net.predict(np.column_stack([G1.ravel(), G2.ravel()])).reshape(G1.shape)
    first, crossings = switches_along(net, np.array([-6.0, -6.0]), np.array([6.0, 6.0]))
    print(f"  along the line from (-6, -6) to (6, 6), through the three cluster centers: starts as Class {first}, "
          f"changes class {len(crossings)} times, at {crossings}")
    print(f"  Class 1 covers {region.mean():.2%} of the plotted grid")
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.contourf(G1, G2, region, levels=[-0.5, 0.5, 1.5], colors=["#cfe0f3", "#fde0c5"])
    scatter_classes(ax, X_te, y_te, [f"{name} (test)" for name in LABELS], size=16, alpha=0.9,
                    edgecolors="white", linewidths=0.3)
    ax.plot(x1, -(w[0] * x1 + b) / w[1], color="black", ls="--", lw=1.3,
            label=f"logistic-regression boundary (test accuracy {base_acc:.2%})")
    handles, names = ax.get_legend_handles_labels()
    handles = [Patch(color="#cfe0f3", label="MLP region: Class 0"),
               Patch(color="#fde0c5", label="MLP region: Class 1")] + handles
    ax.set(title=f"Figure 3 — Exercise 2: MLP decision regions over the {len(X_te)} test points\n"
                 f"MLP test accuracy {acc:.2%}", xlabel="$x_1$", ylabel="$x_2$",
           xlim=(x1[0], x1[-1]), ylim=(x2[0], x2[-1]))
    ax.set_aspect("equal")
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2,
              fontsize=9)  # below the axes: covers no data
    save(fig, "fig3.png")

    return {"baseline_accuracy": base_acc, "layers": LAYERS, "accuracy": acc, "n_parameters": net.n_parameters(),
            "final_loss": float(history[-1])}

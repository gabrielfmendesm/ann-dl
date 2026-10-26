"""Exercise 3 — the same network on three classes.

Trains the unchanged MLP of Exercise 2 with a softmax output and categorical
cross-entropy (Figures 4 and 5), fits the logistic-regression baseline, and
measures the cluster structure used to explain the confusions in item D.
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

from common import EPOCHS, LR, SEED, classification_data, save, split
from mlp import MLP, CategoricalCrossEntropy

LAYERS = [4, 16, 3]
N_CLASSES = 3
N_CLUSTERS = 6


def cluster_of_each_sample(X):
    """Index (0-5) of the generating cluster of every row of X.

    With shuffle=False the generator returns the same samples cluster by cluster
    (250 rows each). The shuffle permutes the rows and, with one permutation shared
    by all rows, the feature columns; a row's sorted values are therefore the same
    in both versions and identify it. Cluster k is assigned to class k % 3.
    """
    X_ordered, _ = classification_data(shuffle=False)
    per_cluster = len(X_ordered) // N_CLUSTERS
    key = lambda row: np.sort(row).tobytes()
    position = {key(row): i // per_cluster for i, row in enumerate(X_ordered)}
    assert len(position) == len(X_ordered), "every row must be identified by its sorted values"
    return np.array([position[key(row)] for row in X])


def run():
    print("\n" + "=" * 72 + "\nEXERCISE 3 — three classes\n" + "=" * 72)

    # --- A: data and split -----------------------------------------------------------------------
    X, y = classification_data()
    cluster = cluster_of_each_sample(X)
    X_tr, X_te, y_tr, y_te, k_tr, k_te = split(X, y, cluster)
    print(f"{len(X)} samples, {X.shape[1]} features, class counts {np.bincount(y).tolist()} | "
          f"train {len(X_tr)} {np.bincount(y_tr).tolist()}, test {len(X_te)} {np.bincount(y_te).tolist()}")
    flipped = y != cluster % N_CLASSES
    print(f"clusters: {np.bincount(cluster).tolist()} samples each, cluster k belongs to class k % 3; "
          f"labels differing from their cluster's class (the generator's 1% label noise, flip_y=0.01): "
          f"{int(flipped.sum())} in total, {int(np.sum(y_te != k_te % N_CLASSES))} in the test set")

    # --- B/C: the same MLP class, softmax output and categorical cross-entropy -------------------
    net = MLP(LAYERS, CategoricalCrossEntropy(N_CLASSES), np.random.default_rng(SEED))
    history = net.fit(X_tr, y_tr, EPOCHS, LR)
    pred = net.predict(X_te)
    acc = float(np.mean(pred == y_te))
    print(f"\nMLP {LAYERS}, tanh hidden layer, softmax output, categorical cross-entropy, full-batch gradient descent, "
          f"lr {LR}, {EPOCHS} epochs, {net.n_parameters()} parameters")
    print(f"  training loss: {history[0]:.4f} at epoch 0, {history[500]:.4f} at epoch 500, "
          f"{history[-1]:.4f} at epoch {EPOCHS}")
    print(f"  accuracy: train {np.mean(net.predict(X_tr) == y_tr):.2%}, test {acc:.2%} "
          f"({int(np.sum(pred != y_te))} of {len(y_te)} test samples wrong)")

    baseline = LogisticRegression(max_iter=1000).fit(X_tr, y_tr)
    base_pred = baseline.predict(X_te)
    base_acc = float(np.mean(base_pred == y_te))
    print(f"  logistic-regression baseline (multinomial, one linear score per class): test accuracy {base_acc:.2%}")
    base_correct = np.diag(confusion_matrix(y_te, base_pred, labels=range(N_CLASSES)))
    print(f"    correct test samples per class: {base_correct.tolist()} of "
          f"{np.bincount(y_te).tolist()}; accuracy per generating cluster: "
          + ", ".join(f"cluster {k} {np.mean(base_pred[k_te == k] == y_te[k_te == k]):.2%}" for k in range(N_CLUSTERS)))

    # Figure 4: training loss per epoch
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(np.arange(len(history)), history, color="tab:purple", label=f"training loss (MLP {LAYERS})")
    ax.set(title=f"Figure 4 — Exercise 3: training loss per epoch (lr {LR}, full batch)",
           xlabel="epoch", ylabel="categorical cross-entropy (training set)", xlim=(0, EPOCHS))
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right")
    save(fig, "fig4.png")

    # Figure 5: confusion matrix on the test set
    cm = confusion_matrix(y_te, pred, labels=range(N_CLASSES))
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    image = ax.imshow(cm, cmap="Blues")
    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=13,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    names = [f"Class {k}" for k in range(N_CLASSES)]
    ax.set(title=f"Figure 5 — Exercise 3: confusion matrix on the {len(y_te)} test samples\n"
                 f"test accuracy {acc:.2%}", xlabel="predicted class", ylabel="true class",
           xticks=range(N_CLASSES), yticks=range(N_CLASSES), xticklabels=names, yticklabels=names)
    fig.colorbar(image, ax=ax, label="number of test samples")
    save(fig, "fig5.png")

    # --- D: which pair is confused most, and where in the cluster structure ----------------------
    print(f"\n  confusion matrix (rows = true class, columns = predicted):\n{cm}")
    pairs = {(i, j): int(cm[i, j] + cm[j, i]) for i in range(N_CLASSES) for j in range(i + 1, N_CLASSES)}
    for (i, j), n in pairs.items():
        print(f"  Class {i} <-> Class {j}: {n} test errors "
              f"({cm[i, j]} true {i} predicted {j}, {cm[j, i]} the other way)")
    worst = max(pairs, key=pairs.get)
    print(f"  most confused pair: Class {worst[0]} and Class {worst[1]} "
          f"({pairs[worst]} of {int(np.sum(pred != y_te))} errors)")

    centroids = np.array([X[cluster == k].mean(axis=0) for k in range(N_CLUSTERS)])
    radius = np.array([np.sqrt(np.mean(np.sum((X[cluster == k] - centroids[k]) ** 2, axis=1)))
                       for k in range(N_CLUSTERS)])
    print("  cluster centroids (4D) and RMS radius:")
    for k in range(N_CLUSTERS):
        print(f"    cluster {k} (class {k % 3}): centroid {centroids[k].round(2).tolist()}, RMS radius {radius[k]:.2f}")
    print("  centroid distance between clusters of different classes (ratio = distance / (radius_i + radius_j)):")
    inter = []
    for a in range(N_CLUSTERS):
        for b in range(a + 1, N_CLUSTERS):
            if a % 3 != b % 3:
                d = float(np.linalg.norm(centroids[a] - centroids[b]))
                inter.append((d / (radius[a] + radius[b]), d, a, b))
    for ratio, d, a, b in sorted(inter):
        print(f"    clusters {a}-{b} (classes {a % 3}-{b % 3}): distance {d:.2f}, ratio {ratio:.2f}")

    # A convex region containing the centers of clusters 1 and 4 (Class 1) contains the segment between
    # them: how close does that segment pass to the center of cluster 2 (Class 2)?
    a, b, c = centroids[1], centroids[4], centroids[2]
    t = np.clip((c - a) @ (b - a) / ((b - a) @ (b - a)), 0.0, 1.0)
    gap = float(np.linalg.norm(a + t * (b - a) - c))
    print(f"  segment between the centers of clusters 1 and 4 (Class 1): closest to the center of cluster 2 "
          f"(Class 2) at distance {gap:.2f} (t = {t:.2f}), against cluster 2's RMS radius {radius[2]:.2f}")

    # Where the test errors come from: the generating cluster of every misclassified sample
    wrong = pred != y_te
    print("  test errors by generating cluster (true class -> predicted class):")
    for k in range(N_CLUSTERS):
        sel = wrong & (k_te == k)
        if sel.any():
            moves = {f"{t}->{p}": int(np.sum((y_te[sel] == t) & (pred[sel] == p))) for t in range(3) for p in range(3)
                     if np.any((y_te[sel] == t) & (pred[sel] == p))}
            print(f"    cluster {k} (class {k % 3}): {int(sel.sum())} errors of "
                  f"{int(np.sum(k_te == k))} test samples, {moves}")

    return {"layers": LAYERS, "accuracy": acc, "baseline_accuracy": base_acc, "n_parameters": net.n_parameters(),
            "final_loss": float(history[-1]), "history": history, "confusion": cm, "worst_pair": worst}

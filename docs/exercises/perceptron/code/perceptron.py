"""Single-layer perceptron, written from scratch with NumPy.

This is the only model in the activity. It is written once and reused,
unchanged, by both exercises: Exercise 1 trains it on separable data, and
Exercise 2 trains the same function on overlapping data with the pocket copy
switched on (pocket=True).
"""

import numpy as np


def step(z):
    """Step activation: 1 where z >= 0, 0 otherwise (element-wise)."""
    return np.where(z >= 0, 1, 0)


def predict(X, w, b):
    """Predicted labels y_hat = step(w . x + b), one per row of X."""
    return step(X @ w + b)


def accuracy(X, y, w, b):
    """Fraction of the samples in (X, y) that (w, b) classifies correctly."""
    return float(np.mean(predict(X, w, b) == y))


def init_weights(rng):
    """Initial parameters: w = rng.normal(0, 0.01, size=2) (mean 0, standard deviation 0.01), b = 0.

    The start must not be w = 0: from an all-zero start the learning rate would
    only rescale the weights (Exercise 1, item D).
    """
    return rng.normal(0.0, 0.01, size=2), 0.0


def train(X, y, w0, b0, eta, max_epochs=100, pocket=False):
    """Train the perceptron with the error-driven update rule for 0/1 labels.

    Each epoch visits the samples in the order of X. For every sample:
        y_hat = step(w . x + b)          prediction
        error = y - y_hat                0 if correct, +1 or -1 on a mistake
        w <- w + eta * error * x         (a correct prediction changes nothing)
        b <- b + eta * error
    Training stops after the first epoch with no update, or after max_epochs.

    pocket=True adds the pocket algorithm: after every update the accuracy on
    the full dataset is recomputed and, if it beats the best value seen so far,
    (w, b) is copied into the pocket. The copy is the only thing it adds.
    """
    w = np.array(w0, dtype=float)  # work on a copy: the caller's w0 is reused by other runs
    b = float(b0)

    acc0 = accuracy(X, y, w, b)
    history = {
        "accuracy": [acc0],          # full-dataset accuracy after each epoch (index 0 = before training)
        "updates": [0],              # number of updates made during each epoch
        "weights": [(w.copy(), b)],  # (w, b) at the end of each epoch
        "update_positions": [],      # (epoch, sample index) of every update
    }
    if pocket:
        best = {"w": w.copy(), "b": b, "accuracy": acc0, "epoch": 0}  # the pocket starts with the initial weights
        history["pocket_accuracy"] = [acc0]  # best-so-far accuracy at the end of each epoch

    for epoch in range(1, max_epochs + 1):
        n_updates = 0
        for i in range(len(X)):
            error = y[i] - step(X[i] @ w + b)
            if error != 0:
                w += eta * error * X[i]
                b += eta * error
                n_updates += 1
                history["update_positions"].append((epoch, i))
                if pocket:
                    acc = accuracy(X, y, w, b)
                    if acc > best["accuracy"]:
                        best = {"w": w.copy(), "b": b, "accuracy": acc, "epoch": epoch}
        history["accuracy"].append(accuracy(X, y, w, b))
        history["updates"].append(n_updates)
        history["weights"].append((w.copy(), b))
        if pocket:
            history["pocket_accuracy"].append(best["accuracy"])
        if n_updates == 0:  # a full pass without a single mistake: converged
            break

    return {
        "w": w,
        "b": b,
        "epochs": epoch,
        "converged": n_updates == 0,
        "history": history,
        "pocket": best if pocket else None,
    }

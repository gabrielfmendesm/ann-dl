"""Exercise 4 — does a second hidden layer help?

A controlled comparison on the Exercise 3 data: the only change is depth
([4, 16, 3] against [4, 16, 16, 3]). Both networks get the same data split,
learning rate, epochs, loss and code, and every seed starts both networks from
np.random.default_rng(seed). The comparison is repeated over SEEDS (Figure 6).
"""

import matplotlib.pyplot as plt
import numpy as np

from common import EPOCHS, LR, SEED, classification_data, save, split
from mlp import MLP, CategoricalCrossEntropy

SHALLOW = [4, 16, 3]        # the Exercise 3 network
DEEP = [4, 16, 16, 3]       # one more hidden layer of the same width
SEEDS = list(range(SEED, SEED + 10))   # 42, 43, ..., 51


def train(layers, seed, X_tr, y_tr, X_te, y_te):
    net = MLP(layers, CategoricalCrossEntropy(3), np.random.default_rng(seed))
    history = net.fit(X_tr, y_tr, EPOCHS, LR)
    return {"history": history, "train_loss": float(history[-1]), "n_parameters": net.n_parameters(),
            "train_accuracy": float(np.mean(net.predict(X_tr) == y_tr)),
            "test_accuracy": float(np.mean(net.predict(X_te) == y_te))}


def run(ex3):
    print("\n" + "=" * 72 + "\nEXERCISE 4 — one hidden layer against two\n" + "=" * 72)
    X, y = classification_data()
    X_tr, X_te, y_tr, y_te = split(X, y)

    runs = {name: [train(layers, s, X_tr, y_tr, X_te, y_te) for s in SEEDS]
            for name, layers in (("shallow", SHALLOW), ("deep", DEEP))}
    # The seed-42 shallow run is the Exercise 3 network itself
    assert runs["shallow"][0]["test_accuracy"] == ex3["accuracy"]
    assert np.allclose(runs["shallow"][0]["history"], ex3["history"])

    deep_42 = runs["deep"][0]
    print(f"A — deeper network {DEEP}, seed {SEED}, lr {LR}, {EPOCHS} epochs: {deep_42['n_parameters']} parameters, "
          f"final training loss {deep_42['train_loss']:.4f}, test accuracy {deep_42['test_accuracy']:.2%} "
          f"(Exercise 3, same seed: {ex3['accuracy']:.2%})")

    print(f"\nB — seeds {SEEDS[0]}-{SEEDS[-1]} ({len(SEEDS)} seeds), per seed (shallow / deep):")
    for s, a, b in zip(SEEDS, runs["shallow"], runs["deep"]):
        print(f"  seed {s}: test {a['test_accuracy']:.2%} / {b['test_accuracy']:.2%} | "
              f"final train loss {a['train_loss']:.4f} / {b['train_loss']:.4f}")
    summary = {}
    for name, layers in (("shallow", SHALLOW), ("deep", DEEP)):
        r = runs[name]
        test = np.array([x["test_accuracy"] for x in r])
        loss = np.array([x["train_loss"] for x in r])
        train_acc = np.array([x["train_accuracy"] for x in r])
        curves = np.array([x["history"] for x in r])
        summary[name] = {"layers": layers, "hidden": len(layers) - 2, "n_parameters": r[0]["n_parameters"],
                         "test_mean": test.mean(), "test_std": test.std(ddof=1), "loss_mean": loss.mean(),
                         "loss_std": loss.std(ddof=1), "train_acc_mean": train_acc.mean(), "curves": curves}
        rises = np.diff(curves, axis=1) > 0                       # epochs where an update raised the training loss
        summary[name]["rising_epochs"] = rises.sum(axis=1)
        summary[name]["largest_rise"] = max(0.0, float(np.max(np.diff(curves, axis=1))))  # 0 when the loss never rose
        early = curves[:, [50, 100, 200, 500]].mean(axis=0)
        print(f"  {name:7s} {layers}: {len(layers) - 2} hidden layer(s), {r[0]['n_parameters']} parameters | "
              f"final train loss {loss.mean():.4f} ± {loss.std(ddof=1):.4f} | train accuracy {train_acc.mean():.2%} | "
              f"test accuracy {test.mean():.2%} ± {test.std(ddof=1):.2%} (min {test.min():.2%}, max {test.max():.2%})")
        print("          mean training loss at epochs 50 / 100 / 200 / 500: " + " / ".join(f"{v:.4f}" for v in early))
        print("          epochs in which the update raised the training loss: "
              f"{summary[name]['rising_epochs'].tolist()} "
              f"(per seed); largest single rise {summary[name]['largest_rise']:.4f}")
    diff = np.array([b["test_accuracy"] - a["test_accuracy"] for a, b in zip(runs["shallow"], runs["deep"])])
    gap = (summary["deep"]["train_acc_mean"] - summary["deep"]["test_mean"],
           summary["shallow"]["train_acc_mean"] - summary["shallow"]["test_mean"])
    print(f"  deep - shallow test accuracy: mean {diff.mean():+.2%}, standard deviation {diff.std(ddof=1):.2%} "
          f"(standard error {diff.std(ddof=1) / np.sqrt(len(diff)):.2%}); deep ahead in {int(np.sum(diff > 0))}, "
          f"tied in {int(np.sum(diff == 0))}, behind in {int(np.sum(diff < 0))} of {len(diff)} seeds")
    print(f"  train - test accuracy gap: shallow {gap[1]:.2%}, deep {gap[0]:.2%}")

    # Figure 6: both training-loss curves on the same axes, every seed thin, the mean thick
    fig, ax = plt.subplots(figsize=(10, 5.5))
    epochs = np.arange(EPOCHS + 1)
    for name, color in (("shallow", "tab:blue"), ("deep", "tab:red")):
        s = summary[name]
        for curve in s["curves"]:
            ax.plot(epochs, curve, color=color, lw=0.6, alpha=0.25)
        ax.plot(epochs, s["curves"].mean(axis=0), color=color, lw=2.2,
                label=f"{'Exercise 3' if name == 'shallow' else 'Exercise 4'} {s['layers']}: "
                      f"mean of {len(SEEDS)} seeds "
                      f"(final {s['loss_mean']:.4f})")
        ax.plot([], [], color=color, lw=0.6, alpha=0.5, label=f"{s['layers']}: individual seeds")
    ax.set(title=f"Figure 6 — Training loss per epoch: one hidden layer against two\n"
                 f"same data, lr {LR}, {EPOCHS} epochs, seeds {SEEDS[0]}-{SEEDS[-1]}",
           xlabel="epoch", ylabel="categorical cross-entropy (training set)", xlim=(0, EPOCHS), ylim=(0, 1.3))
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    save(fig, "fig6.png")

    return {"deep_layers": DEEP, "deep_accuracy": deep_42["test_accuracy"], "seeds": SEEDS, "summary": summary,
            "diff_mean": float(diff.mean()), "diff_std": float(diff.std(ddof=1))}

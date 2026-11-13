"""Steps 2 to 4 — the network, its training with early stopping, regularization and the configuration search.

Everything here is measured on the validation folds of the training rows; the test set is not touched.
Tables 3-7, Figures 1-2.
"""

import itertools
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import N_FOLDS, NEUTRAL, Results, save_figure
from network import BATCH_SIZE, MAX_EPOCHS, MLP, PATIENCE
from validation import cv_network, fold_matrices, mean_sd

DEFAULT = {"hidden": [32, 16], "dropout": 0.0, "lr": 1e-3, "weight_decay": 0.0}   # the network of step 2
REGULARIZATION = {                       # name: (weight_decay, dropout)
    "none": (0.0, 0.0),
    "weight decay 0.1": (0.1, 0.0),
    "weight decay 1": (1.0, 0.0),
    "weight decay 10": (10.0, 0.0),
    "dropout 0.1": (0.0, 0.1),
    "dropout 0.3": (0.0, 0.3),
    "dropout 0.5": (0.0, 0.5),
    "weight decay 1 + dropout 0.3": (1.0, 0.3),
}
LEARNING_RATES = (1e-3, 3e-3)
ARCHITECTURES = ([16], [32, 16], [64, 32])
NONE_COLOR = "#eb6834"
WEIGHT_DECAY_COLORS = ["#9ec5f0", "#2a78d6", "#0d3b7a"]    # light to dark: a stronger decay
DROPOUT_COLORS = ["#9be0c4", "#1baf7a", "#0b6b47"]         # light to dark: a higher rate
FOLD_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]


def with_regularization(config, name):
    weight_decay, dropout = REGULARIZATION[name]
    return {**config, "weight_decay": weight_decay, "dropout": dropout}


def regularization_name(config):
    return next(name for name, (wd, p) in REGULARIZATION.items()
                if (wd, p) == (config["weight_decay"], config["dropout"]))


def layers(config):
    return "-".join(map(str, config["hidden"]))


def adult_low_bmi_to_missing(X):
    """The EDA's sensitivity check: the adult BMIs below 12, probable entry errors, set to missing."""
    X = X.copy()
    X.loc[(X["age"] >= 18) & (X["bmi"] < 12), "bmi"] = np.nan
    return X


def summary_row(cv):
    (ap, ap_sd), auc = mean_sd(cv["ap"]), np.mean(cv["auc"])
    return {"AP (mean)": ap, "AP (sd)": ap_sd, "ROC-AUC (mean)": auc,
            "best epoch (median)": int(np.median(cv["best_epoch"]))}


def architecture_table(config, n_inputs):
    """Layer by layer: shapes, activation, initialization and parameters."""
    rows, width = [], n_inputs
    for i, units in enumerate(config["hidden"], start=1):
        rows.append({"layer": f"hidden {i}", "shape": f"{width} → {units}", "activation": "ReLU",
                     "initialization": "He (Kaiming) normal", "parameters": width * units + units})
        width = units
    rows.append({"layer": "output", "shape": f"{width} → 1", "activation": "none (logit)",
                 "initialization": "Glorot (Xavier) normal", "parameters": width + 1})
    table = pd.DataFrame(rows)
    assert table["parameters"].sum() == MLP(n_inputs, config["hidden"]).n_parameters()
    return table


def plot_curves(cv):
    """Figure 1: training and validation loss, and validation AP, per epoch for every fold."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for k, history in enumerate(cv["histories"]):
        epochs = np.arange(1, len(history["val_loss"]) + 1)
        color, best = FOLD_COLORS[k], cv["best_epoch"][k]
        axes[0].plot(epochs, history["train_loss"], color=color, lw=1, ls="--")
        axes[0].plot(epochs, history["val_loss"], color=color, lw=1.4, label=f"fold {k + 1}")
        axes[0].plot(best, history["val_loss"][best - 1], "o", color=color, ms=5)
        axes[1].plot(epochs, history["val_ap"], color=color, lw=1.4, label=f"fold {k + 1}")
        axes[1].plot(best, history["val_ap"][best - 1], "o", color=color, ms=5)
    axes[0].plot([], [], color=NEUTRAL, ls="--", label="training loss")
    axes[0].plot([], [], "o", color=NEUTRAL, label="best epoch (restored)")
    axes[0].set(title="Loss per epoch: validation (solid), training (dashed)", xlabel="epoch",
                ylabel="weighted binary cross-entropy")
    axes[1].plot([], [], "o", color=NEUTRAL, label="best epoch (restored)")
    axes[1].set(title="Validation average precision per epoch", xlabel="epoch", ylabel="average precision")
    axes[0].legend(ncol=2)
    axes[1].legend(ncol=2)
    save_figure(fig, 1, "The network of step 2 trained with early stopping, one curve per validation fold")


def plot_regularization(runs):
    """Figure 2: fold 1, training (dashed) and validation (solid) loss, each technique at its three strengths next to
    the network without regularization."""
    panels = {"Weight decay": dict(zip([n for n in REGULARIZATION if n.startswith("weight decay") and "+" not in n],
                                       WEIGHT_DECAY_COLORS)),
              "Dropout": dict(zip([n for n in REGULARIZATION if n.startswith("dropout")], DROPOUT_COLORS))}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, (technique, colors) in zip(axes, panels.items()):
        for name, color in {"none": NONE_COLOR, **colors}.items():
            history, best = runs[name]["histories"][0], runs[name]["best_epoch"][0]
            epochs = np.arange(1, len(history["val_loss"]) + 1)
            ax.plot(epochs, history["train_loss"], color=color, lw=1, ls="--")
            ax.plot(epochs, history["val_loss"], color=color, lw=1.6, label=name)
            ax.plot(best, history["val_loss"][best - 1], "o", color=color, ms=5)
        ax.plot([], [], color=NEUTRAL, ls="--", label="training loss (dashed)")
        ax.plot([], [], "o", color=NEUTRAL, label="best epoch (restored)")
        ax.set(title=f"{technique}: validation (solid) and training (dashed) loss, fold 1", xlabel="epoch")
        ax.legend(title="regularization")
    axes[0].set_ylabel("weighted binary cross-entropy")
    save_figure(fig, 2, f"The network of step 2 ({layers(DEFAULT)}, lr {DEFAULT['lr']:g}) with each technique at "
                        "three strengths")


def run(X_train, y_train, folds, make_preprocessor, mats, n_inputs):
    """Steps 2 to 4 on the fold matrices `mats` (Tables 3-7, Figures 1-2). Returns the chosen configuration, its
    out-of-fold probabilities and the number of epochs of the final retraining."""
    results = Results("search")

    # --- Step 2: the network ---------------------------------------------------------------------
    results.table(3, f"The network of step 2, layer by layer ({n_inputs} inputs)",
                  architecture_table(DEFAULT, n_inputs))
    n_params = MLP(n_inputs, DEFAULT["hidden"]).n_parameters()
    results.metrics.update({
        "n_inputs": n_inputs, "fold_inputs": [fold.X_train.shape[1] for fold in mats], "default_params": n_params,
        "train_rows": len(y_train), "fold_train_rows": [len(fold.y_train) for fold in mats], "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS, "patience": PATIENCE,
        "pos_weight_folds": [float((fold.y_train == 0).sum() / fold.y_train.sum()) for fold in mats],
    })

    # --- Step 3: the network trained with early stopping -----------------------------------------
    default_cv = cv_network(DEFAULT, mats)
    rows = []
    for k, history in enumerate(default_cv["histories"]):
        best = default_cv["best_epoch"][k]
        rows.append({"fold": k + 1, "validation positives": int(mats[k].y_val.sum()),
                     "epochs run": len(history["val_loss"]), "best epoch": best,
                     "training loss": history["train_loss"][best - 1], "validation loss": history["val_loss"][best - 1],
                     "validation AP": default_cv["ap"][k]})
    results.table(4, "Early stopping of the network of step 2 on each fold (losses and AP at the restored epoch)",
                  pd.DataFrame(rows))
    results.metrics["default"] = {
        "ap": default_cv["ap"], "auc": default_cv["auc"], "best_epoch": default_cv["best_epoch"],
        "epochs_run": [len(h["val_loss"]) for h in default_cv["histories"]],
        "train_loss_last": [h["train_loss"][-1] for h in default_cv["histories"]],
        "val_loss_last": [h["val_loss"][-1] for h in default_cv["histories"]],
        "val_loss_best": [h["val_loss"][b - 1] for h, b in zip(default_cv["histories"], default_cv["best_epoch"])],
        # the validation AP after the restored epoch: its maximum over the run, the epoch of that maximum, and its last
        # value, to compare what the loss and the ranking say about when to stop
        "val_ap_max": [max(h["val_ap"]) for h in default_cv["histories"]],
        "val_ap_max_epoch": [int(np.argmax(h["val_ap"])) + 1 for h in default_cv["histories"]],
        "val_ap_last": [h["val_ap"][-1] for h in default_cv["histories"]],
    }
    plot_curves(default_cv)

    # --- Step 4: regularization, then the search -------------------------------------------------
    cache = {}

    def cv(config):
        key = (config["lr"], tuple(config["hidden"]), config["weight_decay"], config["dropout"])
        if key not in cache:
            cache[key] = default_cv if config == DEFAULT else cv_network(config, mats)
        return cache[key]

    reg_runs = {name: cv(with_regularization(DEFAULT, name)) for name in REGULARIZATION}
    rows = []
    steps_per_epoch = math.ceil(min(len(fold.y_train) for fold in mats) / BATCH_SIZE)
    for name, run_ in reg_runs.items():
        gap = np.mean([h["val_loss"][b - 1] - h["train_loss"][b - 1]
                       for h, b in zip(run_["histories"], run_["best_epoch"])])
        # AdamW multiplies every weight by (1 - lr * weight_decay) at each step: the factor accumulated by the
        # median restored epoch shows how strongly the decay acted
        decay = (1 - DEFAULT["lr"] * REGULARIZATION[name][0]) ** (steps_per_epoch * np.median(run_["best_epoch"]))
        rows.append({"regularization": name, **summary_row(run_), "validation − training loss": gap,
                     "weights kept by the decay": decay})
    results.table(5, f"The network of step 2 ({layers(DEFAULT)}, lr {DEFAULT['lr']:g}) with each regularization "
                     f"({N_FOLDS} validation folds; the loss gap is at the restored epoch)", pd.DataFrame(rows))
    results.metrics["steps_per_epoch"] = steps_per_epoch
    plot_regularization(reg_runs)

    configs = [with_regularization({"hidden": list(h), "lr": lr, "dropout": 0.0, "weight_decay": 0.0}, reg)
               for lr, h, reg in itertools.product(LEARNING_RATES, ARCHITECTURES, REGULARIZATION)]
    rows = []
    for config in configs:
        rows.append({"learning rate": config["lr"], "hidden layers": layers(config),
                     "regularization": regularization_name(config),
                     "parameters": MLP(n_inputs, config["hidden"]).n_parameters(), **summary_row(cv(config))})
    # sorted by mean AP; the index keeps each row's position in `configs`
    search = pd.DataFrame(rows).sort_values("AP (mean)", ascending=False, kind="stable")
    search.insert(0, "rank", np.arange(1, len(search) + 1))

    # Selection, on validation only: the differences at the top are within the fold-to-fold noise, so the one-
    # standard-error rule picks the smallest network whose mean AP is within one standard error of the best one
    # (among networks of the same size, the one with the highest mean AP).
    best = search.iloc[0]
    threshold = best["AP (mean)"] - best["AP (sd)"] / np.sqrt(N_FOLDS)
    eligible = search[search["AP (mean)"] >= threshold]
    pick = eligible.sort_values(["parameters", "AP (mean)"], ascending=[True, False], kind="stable").iloc[0]
    search["within one SE"] = np.where(search["AP (mean)"] >= threshold, "yes", "")
    search["chosen"] = np.where(search["rank"] == pick["rank"], "◀", "")
    results.table(6, f"The search: {len(configs)} configurations on the {N_FOLDS} validation folds, sorted by mean "
                     "AP", search, floatfmt={"learning rate": "g"})
    chosen = configs[pick.name]
    chosen_cv = cv(chosen)

    # --- Step 4: the BMI sensitivity check of the EDA, on the chosen configuration ---------------
    low_bmi = cv_network(chosen, fold_matrices(X_train, y_train, folds, make_preprocessor, adult_low_bmi_to_missing))
    n_low_bmi = int(((X_train["age"] >= 18) & (X_train["bmi"] < 12)).sum())
    results.table(7, f"The chosen configuration with the adult BMIs below 12 set to missing ({N_FOLDS} validation "
                     "folds)",
                  pd.DataFrame([{"data": "as recorded", **summary_row(chosen_cv)},
                                {"data": f"the {n_low_bmi} adult BMIs below 12 set to missing",
                                 **summary_row(low_bmi)}]))

    results.metrics.update({
        "n_configs": len(configs), "best_rank1": best.to_dict(), "one_se_threshold": threshold,
        "n_within_one_se": int(len(eligible)), "chosen": chosen, "chosen_rank": int(pick["rank"]),
        "chosen_params": int(pick["parameters"]), "chosen_label": f"{layers(chosen)}, {pick['regularization']}",
        "chosen_cv": {"ap": chosen_cv["ap"], "auc": chosen_cv["auc"], "best_epoch": chosen_cv["best_epoch"]},
        "final_epochs": int(np.median(chosen_cv["best_epoch"])), "low_bmi_ap": low_bmi["ap"], "n_low_bmi": n_low_bmi,
    })
    results.write()
    print(search.head(10).to_string(index=False))
    print(f"chosen: rank {int(pick['rank'])}, {results.metrics['chosen_label']}, lr {chosen['lr']:g}; "
          f"final epochs {results.metrics['final_epochs']}")
    return {"config": chosen, "oof": chosen_cv["oof"], "epochs": results.metrics["final_epochs"]}

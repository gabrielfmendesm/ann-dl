"""Step 5 — the test evaluation, done once.

Everything that is decided here is decided on validation first: each model's decision threshold comes from its
out-of-fold validation probabilities and the cost of the errors fixed in step 0. Only then are the chosen network
(retrained on all training rows, with each seed) and the baselines (refitted on all training rows) applied to the
test rows, once. Tables 8-11, Figures 3-5.
"""

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import (average_precision_score, precision_recall_curve, precision_recall_fscore_support,
                             roc_auc_score, roc_curve)

from baselines import make_baselines
from common import COST_MISS, CRITERION_GAIN, MODEL_COLORS, NEUTRAL, SEED, SEEDS, Results, save_figure
from network import predict_proba, train
from validation import mean_sd

COST_SENSITIVITY = (10, COST_MISS, 50)   # the threshold under other costs of a miss, on validation only
N_BOOTSTRAP = 2000                        # resamples of the test rows for the paired bootstrap
LEARNED_BASELINES = ("Logistic regression", "Gradient boosting")
COST_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]   # one per cost of COST_SENSITIVITY; the chosen one in orange


def choose_threshold(y, p, cost_miss):
    """The probability threshold (flag when p >= t) with the lowest total cost cost_miss * FN + FP on (y, p).
    Candidates are the observed probabilities and +inf (flag nobody); ties go to the highest threshold."""
    candidates = np.append(np.unique(p), np.inf)
    costs = np.array([cost_miss * np.sum((p < t) & (y == 1)) + np.sum((p >= t) & (y == 0)) for t in candidates])
    best = np.flatnonzero(costs == costs.min()).max()
    return float(candidates[best]), costs, candidates


def decision_metrics(y, p, threshold, cost_miss=COST_MISS):
    """The confusion counts, precision, recall and F1 of both classes, macro F1 and the cost per patient."""
    flag = p >= threshold
    tn, fp = int(np.sum(~flag & (y == 0))), int(np.sum(flag & (y == 0)))
    fn, tp = int(np.sum(~flag & (y == 1))), int(np.sum(flag & (y == 1)))
    precision, recall, f1, _ = precision_recall_fscore_support(y, flag.astype(int), labels=[0, 1], zero_division=0)
    return {"TN": tn, "FP": fp, "FN": fn, "TP": tp, "precision": precision.tolist(), "recall": recall.tolist(),
            "f1": f1.tolist(), "macro_f1": float(np.mean(f1)), "flagged": int(flag.sum()),
            "cost_per_patient": (cost_miss * fn + fp) / len(y)}


def plot_threshold(y, p, chosen):
    """Figure 3: validation cost per patient against the threshold, for three costs of a miss."""
    fig, ax = plt.subplots(figsize=(9, 4.6))
    for cost, color, lw in zip(COST_SENSITIVITY, COST_COLORS, [1.2, 2.2, 1.2]):
        t, costs, candidates = choose_threshold(y, p, cost)
        finite = np.isfinite(candidates)
        ax.plot(candidates[finite], costs[finite] / len(y), color=color, lw=lw,
                label=f"a miss costs {cost} false alarms (best threshold {t:.3f})")
        ax.plot(t, costs[candidates == t][0] / len(y), "o", color=color, ms=6)
    ax.axvline(chosen, color=NEUTRAL, ls=":", lw=1, label=f"chosen threshold {chosen:.3f} (cost {COST_MISS})")
    ax.set(title="Out-of-fold validation probabilities of the chosen network",
           xlabel="threshold on the predicted probability", ylabel="cost per patient (false alarm = 1)", xlim=(0, 1))
    ax.legend()
    save_figure(fig, 3, "Choosing the decision threshold on validation from the cost of the errors")


def plot_test_curves(y, probabilities, metrics, labels):
    """Figure 4: precision-recall and ROC curves on the test set, with each model's operating point."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for name, p in probabilities.items():
        color, m, label = MODEL_COLORS[name], metrics[name], labels.get(name, name)
        precision, recall, _ = precision_recall_curve(y, p)
        fpr, tpr, _ = roc_curve(y, p)
        axes[0].step(recall, precision, where="post", color=color, lw=1.6, label=f"{label} (AP {m['ap']:.3f})")
        axes[1].plot(fpr, tpr, color=color, lw=1.6, label=f"{label} (ROC-AUC {m['auc']:.3f})")
        if m["TP"] + m["FP"] > 0:   # precision is undefined for a model that flags nobody
            axes[0].plot(m["recall"][1], m["precision"][1], "o", color=color, ms=6)
        axes[1].plot(m["FP"] / (m["FP"] + m["TN"]), m["recall"][1], "o", color=color, ms=6)
    # no prevalence line or diagonal is drawn: the majority class, which gives every patient the same score, traces both
    axes[0].set(title="Precision–recall (dots: the validation threshold)", xlabel="recall (strokes found)",
                ylabel="precision (alarms that are strokes)", xlim=(0, 1), ylim=(0, 1))
    axes[1].set(title="ROC (dots: the validation threshold)", xlabel="false-positive rate",
                ylabel="true-positive rate")
    axes[0].legend(fontsize=7.5)
    axes[1].legend(fontsize=7.5, loc="lower right")
    save_figure(fig, 4, "The network and the baselines on the test set")


def plot_confusion(matrices):
    """Figure 5: confusion matrices on the test set at the validation thresholds."""
    fig, axes = plt.subplots(1, len(matrices), figsize=(4.4 * len(matrices), 4.2))
    for ax, (name, m) in zip(axes, matrices.items()):
        cm = np.array([[m["TN"], m["FP"]], [m["FN"], m["TP"]]])
        image = ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.sum())   # the same color scale in every panel
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", fontsize=13,
                        color="white" if cm[i, j] > cm.sum() / 2 else "black")
        ax.set_xticks([0, 1], ["no stroke", "stroke"])
        ax.set_yticks([0, 1], ["no stroke", "stroke"])
        ax.set(title=f"{name}\nthreshold {m['threshold']:.3f}", xlabel="predicted", ylabel="actual")
        ax.grid(False)
        fig.colorbar(image, ax=ax, fraction=0.046, label="test patients")
    save_figure(fig, 5, "Confusion matrices on the test set at the validation thresholds")


def paired_bootstrap(y, network_seeds, baselines):
    """Resample the test rows with replacement, the same rows for every model; on each resample, the AP of the network
    (the seed-42 model and the mean over the seeds) and its gap to each baseline. A resample without a positive has
    no AP and is skipped."""
    rng = np.random.default_rng(SEED)
    one_seed = f"seed {SEED}"
    ap = {one_seed: [], "mean of the seeds": []}
    diffs = {(net, name): [] for net in ap for name in baselines}
    for _ in range(N_BOOTSTRAP):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].sum() == 0:
            continue
        seeds = [average_precision_score(y[idx], p[idx]) for p in network_seeds.values()]
        nets = {one_seed: seeds[list(network_seeds).index(SEED)], "mean of the seeds": float(np.mean(seeds))}
        for net, value in nets.items():
            ap[net].append(value)
            for name, p in baselines.items():
                diffs[(net, name)].append(value - average_precision_score(y[idx], p[idx]))
    return {"n": len(ap[one_seed]),
            "network_ap_ci": {k: np.percentile(v, [2.5, 97.5]).tolist() for k, v in ap.items()},
            "diff": {f"{net} - {name}": {"ci": np.percentile(d, [2.5, 97.5]).tolist(),
                                         "share_positive": float(np.mean(np.array(d) > 0))}
                     for (net, name), d in diffs.items()}}


def mcnemar(y, flag_a, flag_b):
    """Exact McNemar test on the patients the two models classify differently: under the null both kinds of
    disagreement are equally likely, so the count of one kind is Binomial(disagreements, 1/2)."""
    right_a, right_b = flag_a == (y == 1), flag_b == (y == 1)
    only_a, only_b = int(np.sum(right_a & ~right_b)), int(np.sum(~right_a & right_b))
    p = binomtest(only_a, only_a + only_b, 0.5).pvalue if only_a + only_b else 1.0
    return {"only_first_right": only_a, "only_second_right": only_b, "p": float(p)}


def run(X_train, X_test, y_train, y_test, make_preprocessor, chosen, oof_baselines, best_baseline):
    """Step 5 (Tables 8-11, Figures 3-5). Returns what step 6 analyzes: the seed-42 network, the fitted pipeline, the
    test probabilities of every model and the network's threshold."""
    results = Results("evaluation")

    # --- On validation: the thresholds -----------------------------------------------------------
    oof = {"Network": chosen["oof"], **oof_baselines}
    # The same rule for every model. The majority class gives every patient the same score, so its only choices are
    # flagging everyone or no one; the rule picks the cheaper of the two on validation.
    thresholds = {name: choose_threshold(y_train, p, COST_MISS)[0] for name, p in oof.items()}
    rows = []
    for cost in COST_SENSITIVITY:
        t = choose_threshold(y_train, chosen["oof"], cost)[0]
        m = decision_metrics(y_train, chosen["oof"], t, cost)
        rows.append({"cost of a miss": cost, "threshold": t, "recall (1)": m["recall"][1],
                     "precision (1)": m["precision"][1], "flagged (% of patients)": 100 * m["flagged"] / len(y_train),
                     "cost per patient": m["cost_per_patient"]})
    results.table(8, "The chosen network's threshold on the out-of-fold validation probabilities, for three costs "
                     "of a miss", pd.DataFrame(rows),
                  floatfmt={"cost of a miss": "d", "flagged (% of patients)": ".2f"})
    plot_threshold(y_train, chosen["oof"], thresholds["Network"])
    # pos_weight multiplies the odds the network learns by w = negatives / positives; dividing the odds of the threshold
    # by w gives the probability it stands for once that factor is undone, to compare with p* = 1 / (1 + COST_MISS)
    w = (y_train == 0).sum() / (y_train == 1).sum()
    t = thresholds["Network"]
    results.metrics.update({
        "thresholds": {k: (v if np.isfinite(v) else None) for k, v in thresholds.items()}, "pos_weight": float(w),
        "threshold_corrected": t / (t + (1 - t) * w), "p_star": 1 / (1 + COST_MISS),
        "validation_decisions": {name: decision_metrics(y_train, p, thresholds[name]) for name, p in oof.items()},
    })

    # --- Once: the final models on the test rows -------------------------------------------------
    preprocessor = make_preprocessor().fit(X_train)
    X_train_t, X_test_t = preprocessor.transform(X_train), preprocessor.transform(X_test)
    seed_models = {seed: train(X_train_t, y_train, chosen["config"], seed, epochs=chosen["epochs"])[0]
                   for seed in SEEDS}
    seed_probs = {seed: predict_proba(model, X_test_t) for seed, model in seed_models.items()}
    probabilities = {"Network": seed_probs[SEED]}
    for name, model in make_baselines().items():
        probabilities[name] = model.fit(X_train_t, y_train).predict_proba(X_test_t)[:, 1]

    metrics = {}
    for name, p in probabilities.items():
        metrics[name] = decision_metrics(y_test, p, thresholds[name])
        metrics[name].update({"ap": average_precision_score(y_test, p), "auc": roc_auc_score(y_test, p),
                              "threshold": thresholds[name] if np.isfinite(thresholds[name]) else None})
    per_seed = {seed: {**decision_metrics(y_test, p, thresholds["Network"]), "ap": average_precision_score(y_test, p),
                       "auc": roc_auc_score(y_test, p)} for seed, p in seed_probs.items()}

    def mean_over_seeds(key, index=None):
        return float(np.mean([m[key] if index is None else m[key][index] for m in per_seed.values()]))

    labels = {"Network": f"Network (seed {SEED})"}
    rows, per_class = [], []
    for name, m in metrics.items():
        rows.append({"model": labels.get(name, name), "AP": m["ap"], "ROC-AUC": m["auc"],
                     "threshold (validation)": m["threshold"] if m["threshold"] is not None else np.nan,
                     "TN": m["TN"], "FP": m["FP"], "FN": m["FN"], "TP": m["TP"],
                     "cost per patient": m["cost_per_patient"]})
        per_class.append({"model": labels.get(name, name), "precision (0)": m["precision"][0],
                          "recall (0)": m["recall"][0],
                          "F1 (0)": m["f1"][0], "precision (1)": m["precision"][1], "recall (1)": m["recall"][1],
                          "F1 (1)": m["f1"][1], "macro F1": m["macro_f1"]})
        if name == "Network":   # the same row averaged over the seeds, right below the seed-42 one
            rows.append({"model": f"Network (mean of {len(SEEDS)} seeds)", "AP": mean_over_seeds("ap"),
                         "ROC-AUC": mean_over_seeds("auc"), "threshold (validation)": thresholds["Network"],
                         **{k: mean_over_seeds(k) for k in ("TN", "FP", "FN", "TP")},
                         "cost per patient": mean_over_seeds("cost_per_patient")})
            per_class.append({"model": f"Network (mean of {len(SEEDS)} seeds)",
                              **{f"{metric} ({c})": mean_over_seeds(key, c) for c in (0, 1)
                                 for metric, key in (("precision", "precision"), ("recall", "recall"), ("F1", "f1"))},
                              "macro F1": mean_over_seeds("macro_f1")})
    counts = {k: ".1f" for k in ("TN", "FP", "FN", "TP")}   # the seed mean is not a whole number of patients
    # dtype=object keeps each model's whole counts as integers next to the seed means
    results.table(9, "The test set, used once: AP, ROC-AUC and the decisions at each model's validation threshold",
                  pd.DataFrame(rows, dtype=object), floatfmt=counts)
    columns = ["model", "precision (0)", "recall (0)", "F1 (0)", "precision (1)", "recall (1)", "F1 (1)", "macro F1"]
    results.table(10, "Precision, recall and F1 per class on the test set (0 = no stroke, 1 = stroke)",
                  pd.DataFrame(per_class)[columns])

    seeds = pd.DataFrame([{"seed": seed, "AP": m["ap"], "ROC-AUC": m["auc"], "recall (1)": m["recall"][1],
                           "false alarms": m["FP"], "cost per patient": m["cost_per_patient"]}
                          for seed, m in per_seed.items()])
    (ap_mean, ap_sd), (auc_mean, auc_sd) = mean_sd(seeds["AP"]), mean_sd(seeds["ROC-AUC"])
    results.table(11, f"The chosen network retrained with {len(SEEDS)} seeds, on the test set", seeds,
                  floatfmt={"seed": "d"})

    # --- The success criterion of step 0, and how far the numbers can be trusted -----------------
    base_ap = metrics[best_baseline]["ap"]
    gain = ap_mean / base_ap - 1
    met = bool(gain >= CRITERION_GAIN and ap_mean - base_ap > ap_sd)
    bootstrap = paired_bootstrap(y_test, seed_probs, {k: probabilities[k] for k in LEARNED_BASELINES})
    flags = {name: probabilities[name] >= thresholds[name] for name in ("Network", *LEARNED_BASELINES)}
    cost_rows = {"flag nobody": COST_MISS * y_test.sum() / len(y_test),
                 "flag everyone": (y_test == 0).sum() / len(y_test)}
    results.metrics.update({
        "test": metrics, "per_seed": per_seed,
        "seeds": {"ap": seeds["AP"].tolist(), "auc": seeds["ROC-AUC"].tolist(), "ap_mean": ap_mean, "ap_sd": ap_sd,
                  "auc_mean": auc_mean, "auc_sd": auc_sd},
        "best_baseline": best_baseline, "best_baseline_ap": base_ap, "criterion_gain": gain, "criterion_met": met,
        "gain_vs": {k: ap_mean / metrics[k]["ap"] - 1 for k in LEARNED_BASELINES}, "bootstrap": bootstrap,
        "mcnemar": {k: mcnemar(y_test, flags["Network"], flags[k]) for k in LEARNED_BASELINES},
        "trivial_costs": cost_rows, "test_rows": len(y_test), "test_positives": int(y_test.sum()),
        "epochs": chosen["epochs"],
    })
    plot_test_curves(y_test, probabilities, metrics, labels)
    plot_confusion({labels.get(name, name): metrics[name] for name in ("Network", *LEARNED_BASELINES)})
    results.write()
    print(seeds.to_string(index=False))
    print(f"network AP {ap_mean:.4f} ± {ap_sd:.4f} vs {best_baseline} {base_ap:.4f}: gain {gain:+.2%}, met: {met}")
    print(f"bootstrap: {json.dumps(bootstrap)}")
    print(f"McNemar: {results.metrics['mcnemar']}")
    return {"model": seed_models[SEED], "preprocessor": preprocessor, "probabilities": probabilities,
            "threshold": thresholds["Network"]}

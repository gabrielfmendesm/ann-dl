"""Step 1 — the baselines, on the same pipeline and the same validation folds as the network (Table 2)."""

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from common import N_FOLDS, SEED, Results
from validation import cv_sklearn, mean_sd


def make_baselines():
    """The trivial and the linear baseline the statement requires, and gradient boosting as the optional one.
    The two learned ones weight the classes as the network's pos_weight does."""
    return {
        "Majority class": DummyClassifier(strategy="most_frequent"),
        "Logistic regression": LogisticRegression(class_weight="balanced", max_iter=1000),
        "Gradient boosting": HistGradientBoostingClassifier(class_weight="balanced", random_state=SEED),
    }


def run(mats, y_train):
    """Cross-validate every baseline; returns the out-of-fold probabilities and the best baseline (highest mean AP)."""
    results = Results("baselines")
    rows, oof = [], {}
    for name, model in make_baselines().items():
        cv = cv_sklearn(model, mats)
        oof[name] = cv["oof"]
        (ap, ap_sd), (auc, auc_sd) = mean_sd(cv["ap"]), mean_sd(cv["auc"])
        rows.append({"model": name, "AP (mean)": ap, "AP (sd)": ap_sd, "ROC-AUC (mean)": auc, "ROC-AUC (sd)": auc_sd})
        results.metrics[name] = {"ap_folds": cv["ap"], "auc_folds": cv["auc"]}
    table = pd.DataFrame(rows)
    best = table.loc[table["AP (mean)"].idxmax(), "model"]
    results.metrics.update({"best_baseline": best, "prevalence_train": float(y_train.mean()),
                            "positives_per_fold": [int(fold.y_val.sum()) for fold in mats]})
    results.table(2, f"The baselines on the {N_FOLDS} validation folds (mean and sample standard deviation across "
                     "folds)", table)
    results.write()
    print(table.to_string(index=False))
    print(f"best baseline on validation: {best}")
    return oof, best

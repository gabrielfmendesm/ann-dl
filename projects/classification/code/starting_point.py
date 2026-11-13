"""Step 0 — the starting point: the EDA's pipeline, and the check its synthesis asked for (Table 1).

The EDA found BMI missing about five times more often among the positives and could not tell whether BMI is
recorded before the event the label describes. Its plan: train with and without the BMI-missing indicator, and if
the indicator carries a gain, treat it as a collection artifact and prefer the model without it. This step runs
that check on the validation folds and returns the pipeline every later step uses, for every model alike.
"""

from functools import partial

import pandas as pd

from baselines import make_baselines
from common import N_FOLDS, Results, build_preprocessor
from search import DEFAULT
from validation import cv_network, cv_sklearn, fold_matrices, mean_sd, without_indicator


def run(X_train, y_train, folds):
    """Table 1: the learned baselines and the network of step 2 with and without the indicator. Returns the factory
    of the pipeline every later step uses."""
    results = Results("starting_point")
    pipelines = {"with": build_preprocessor, "without": without_indicator}
    mats = {k: fold_matrices(X_train, y_train, folds, make) for k, make in pipelines.items()}
    widths = {k: [fold.X_train.shape[1] for fold in v] for k, v in mats.items()}

    cross_validate = {name: partial(cv_sklearn, model) for name, model in make_baselines().items()
                      if name != "Majority class"}
    cross_validate["Network (step 2)"] = partial(cv_network, DEFAULT)
    rows = []
    for name, cv in cross_validate.items():
        ap = {k: cv(m)["ap"] for k, m in mats.items()}
        (with_mean, with_sd), (without_mean, without_sd) = mean_sd(ap["with"]), mean_sd(ap["without"])
        rows.append({"model": name, "AP with the indicator": with_mean, "sd (with)": with_sd,
                     "AP without it": without_mean, "sd (without)": without_sd,
                     "gain from the indicator": with_mean - without_mean})
    table = pd.DataFrame(rows)
    results.table(1, f"Validation AP with and without the BMI-missing indicator ({N_FOLDS} folds: mean and sample "
                     "sd)", table)

    gain = table.set_index("model")["gain from the indicator"]
    drop = bool(gain["Network (step 2)"] > 0)   # the EDA's rule: a gain means the indicator is suspect
    results.metrics.update({"gain": gain.to_dict(), "drop_indicator": drop, "input_widths": widths})
    results.write()
    print(table.to_string(index=False))
    print(f"drop the BMI-missing indicator: {drop}; inputs per fold {widths}")
    return without_indicator if drop else build_preprocessor

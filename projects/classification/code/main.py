"""Reproduce every table, number and figure of the classification report.

Usage, from the repository root:
    python docs/projects/classification/code/main.py

The steps run in the order of the report. Steps 0 to 4 use only the training rows and their validation folds.
Steps 5 and 6 are the only ones that receive the test rows: step 5 evaluates every model on them once, step 6
analyzes the errors of the seed-42 network on those same rows and changes no choice.
Writes Figures 1-7 to figures/ and the tables (Markdown and CSV) and numbers (JSON) to results/.
"""

import json

import baselines
import errors
import evaluate
import search
import starting_point
from common import RESULTS_DIR, folds, load_data
from validation import fold_matrices


def main():
    X_train, X_test, y_train, y_test = load_data()
    cv_folds = folds(y_train)

    make_preprocessor = starting_point.run(X_train, y_train, cv_folds)                       # step 0
    n_inputs = make_preprocessor().fit_transform(X_train).shape[1]
    mats = fold_matrices(X_train, y_train, cv_folds, make_preprocessor)
    oof_baselines, best_baseline = baselines.run(mats, y_train)                               # step 1
    chosen = search.run(X_train, y_train, cv_folds, make_preprocessor, mats, n_inputs)        # steps 2-4
    final = evaluate.run(X_train, X_test, y_train, y_test, make_preprocessor, chosen,          # step 5
                         oof_baselines, best_baseline)
    errors.run(X_test, y_test, final)                                                        # step 6
    (RESULTS_DIR / "split_sizes.json").write_text(json.dumps({
        "train": len(y_train), "train_positives": int(y_train.sum()), "test": len(y_test),
        "test_positives": int(y_test.sum()), "validation_folds": [len(v) for _, v in cv_folds],
        "validation_positives": [int(y_train[v].sum()) for _, v in cv_folds]}, indent=2) + "\n")


if __name__ == "__main__":
    main()

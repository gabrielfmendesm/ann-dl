"""Cross-validation on the training rows, shared by every model: the network and the baselines see exactly the same
fold matrices, built by a pipeline fitted on each fold's training part only."""

from typing import NamedTuple

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from common import SEED, build_preprocessor
from network import predict_proba, train


class Fold(NamedTuple):
    """One validation fold: the pipeline's output on the fold's training and validation parts, their labels, and the
    positions of the validation rows among the training rows."""
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    val_idx: np.ndarray


def without_indicator():
    """The EDA's pipeline without its BMI-missing indicator."""
    preprocessor = build_preprocessor()
    preprocessor.set_params(missing="drop")
    return preprocessor


def fold_matrices(X_train, y_train, folds, make_preprocessor, prepare=None):
    """One Fold per validation fold. The pipeline is fitted on the fold's training part and only applied to its
    validation part. `prepare` optionally edits the raw rows first (the BMI sensitivity check)."""
    X = prepare(X_train) if prepare else X_train
    mats = []
    for train_idx, val_idx in folds:
        preprocessor = make_preprocessor()
        mats.append(Fold(preprocessor.fit_transform(X.iloc[train_idx]), y_train[train_idx],
                         preprocessor.transform(X.iloc[val_idx]), y_train[val_idx], val_idx))
    return mats


def scores(y, p):
    """Average precision and ROC-AUC of the probabilities p for the labels y."""
    return average_precision_score(y, p), roc_auc_score(y, p)


def cv_sklearn(model, mats):
    """A scikit-learn baseline refitted on every fold; validation AP and ROC-AUC per fold and the out-of-fold
    probabilities (each training row predicted by the model that did not see it)."""
    out = {"ap": [], "auc": [], "oof": np.zeros(sum(len(fold.val_idx) for fold in mats))}
    for fold in mats:
        p = model.fit(fold.X_train, fold.y_train).predict_proba(fold.X_val)[:, 1]
        out["oof"][fold.val_idx] = p
        ap, auc = scores(fold.y_val, p)
        out["ap"].append(ap)
        out["auc"].append(auc)
    return out


def cv_network(config, mats, seed=SEED):
    """One network per fold, early-stopped on the fold's validation part with its best weights restored; the same
    outputs as cv_sklearn, plus the best epochs and the curves."""
    out = {"ap": [], "auc": [], "oof": np.zeros(sum(len(fold.val_idx) for fold in mats)), "best_epoch": [],
           "histories": []}
    for fold in mats:
        model, history, best_epoch = train(fold.X_train, fold.y_train, config, seed, X_val=fold.X_val,
                                           y_val=fold.y_val)
        p = predict_proba(model, fold.X_val)
        out["oof"][fold.val_idx] = p
        ap, auc = scores(fold.y_val, p)
        out["ap"].append(ap)
        out["auc"].append(auc)
        out["best_epoch"].append(best_epoch)
        out["histories"].append(history)
    return out


def mean_sd(values):
    """Mean and sample standard deviation (ddof = 1) across folds or seeds."""
    return float(np.mean(values)), float(np.std(values, ddof=1))

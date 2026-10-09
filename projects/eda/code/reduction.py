"""Stage 4 of the EDA: preprocessing checks, outliers, PCA, t-SNE and UMAP (Figures 9-13).

The pipeline from preprocessing.py is fitted on the training rows only; the test rows are
only transformed, to report their shape and to prove that nothing was learned from them.
The projections use the transformed training matrix; the target only colors the points.
"""

import importlib.metadata
import platform
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, trustworthiness
from sklearn.neighbors import NearestNeighbors
from umap import UMAP

from common import CLASS_COLORS, CLASS_LABELS, GROUP_COLORS, NEUTRAL, UNITS, Results, save_figure, wilson_interval
from preprocessing import NUMERICAL_FEATURES, SEED, SKEWED_FEATURES, build_preprocessor, load_raw, split

PERPLEXITIES = [5, 30, 50]
N_NEIGHBORS = [5, 15, 50]
KNN = 10   # neighborhood size of the class-enrichment and profile-agreement diagnostics


def outliers(X_train, y_train, res):
    """Flag with 1.5×IQR and with the modified z-score (train fences); nothing is removed."""
    m = res.metrics
    rows, flags = [], {}
    for c in NUMERICAL_FEATURES:
        x = X_train[c]
        q1, q3 = x.quantile(.25), x.quantile(.75)
        lo, hi = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
        iqr_flag = (x < lo) | (x > hi)
        med = x.median()
        mad = (x - med).abs().median()
        mz_flag = 0.6745 * (x - med).abs() / mad > 3.5
        flags[c] = iqr_flag
        rows.append({"feature": c, "lower fence": lo, "upper fence": hi, "flagged (1.5×IQR)": int(iqr_flag.sum()),
                     "flagged (%)": iqr_flag.mean() * 100, "stroke rate among flagged (%)":
                     y_train[iqr_flag].mean() * 100 if iqr_flag.any() else 0.0,
                     "flagged (modified z > 3.5)": int(mz_flag.sum())})
    table = pd.DataFrame(rows)
    res.table(13, "Outliers in the training set (fences fitted on train; no row removed)", table,
              {"lower fence": ".2f", "upper fence": ".2f", "flagged (%)": ".2f",
               "stroke rate among flagged (%)": ".2f"})
    any_flag = pd.concat(flags, axis=1).any(axis=1)
    m["outlier_rows"] = int(any_flag.sum())
    m["outlier_rows_pct"] = float(any_flag.mean() * 100)
    m["outlier_rows_positives"] = int(y_train[any_flag].sum())
    m["outlier_rows_positives_share_of_all_pos"] = float(y_train[any_flag].sum() / y_train.sum() * 100)

    # Skewness before and after log1p: the transform is a hypothesis, so it is re-measured
    rows = []
    for c in SKEWED_FEATURES:
        x = X_train[c].dropna()
        rows.append({"feature": c, "skewness (raw)": x.skew(), "skewness (log1p)": np.log1p(x).skew(),
                     "max/median (raw)": x.max() / x.median(),
                     "max/median (log1p)": np.log1p(x.max()) / np.log1p(x.median())})
    res.table(14, "Skewness of the right-skewed features before and after log1p (train, observed values)",
              pd.DataFrame(rows), {k: ".2f" for k in ["skewness (raw)", "skewness (log1p)",
                                                        "max/median (raw)", "max/median (log1p)"]})

    # What plain standardization or min-max scaling would do with the raw tails (scaling choice, 4A). The
    # z-scores use the same median-imputed base as the pipeline, so they compare with max_abs_z_train
    imputed = X_train[SKEWED_FEATURES].fillna(X_train[SKEWED_FEATURES].median())
    m["raw_max_z"] = {c: float((imputed[c].max() - imputed[c].mean()) / imputed[c].std(ddof=0))
                      for c in SKEWED_FEATURES}
    b = X_train.bmi.dropna()
    span = b.max() - b.min()
    m["bmi_minmax_q1_median_q3"] = [float((b.quantile(q) - b.min()) / span) for q in (.25, .5, .75)]


def pipeline_checks(pre, raw, X_train, X_test, Xt, Xv, names, res):
    """4C contract: no NaN, shapes, names, train-only statistics, unseen categories."""
    m = res.metrics
    m["train_shape"], m["test_shape"] = list(Xt.shape), list(Xv.shape)
    m["nan_train"], m["nan_test"] = int(np.isnan(Xt).sum()), int(np.isnan(Xv).sum())
    m["inf_train"], m["inf_test"] = int(np.isinf(Xt).sum()), int(np.isinf(Xv).sum())
    m["feature_names"] = list(names)

    # Fitted parameters equal the training statistics (not the full-file ones)
    num = pre.named_transformers_["num"]
    log = pre.named_transformers_["log"]
    rows = [{"feature": "age", "imputation median": num["impute"].statistics_[0],
             "scaler mean": num["scale"].mean_[0], "scaler std": num["scale"].scale_[0]}]
    for i, c in enumerate(SKEWED_FEATURES):
        rows.append({"feature": f"log1p({c})", "imputation median": log["impute"].statistics_[i],
                     "scaler mean": log["scale"].mean_[i], "scaler std": log["scale"].scale_[i]})
    res.table(15, "Parameters fitted by the pipeline (training rows only)", pd.DataFrame(rows),
              {"imputation median": ".3f", "scaler mean": ".4f", "scaler std": ".4f"})
    assert np.isclose(log["impute"].statistics_[1], X_train.bmi.median())
    # the full-file medians come from the raw file, read once in stage 1, to show what fitting there would give
    m["bmi_median_train"], m["bmi_median_full"] = float(X_train.bmi.median()), float(raw.bmi.median())
    m["glucose_median_train"], m["glucose_median_full"] = (float(X_train.avg_glucose_level.median()),
                                                           float(raw.avg_glucose_level.median()))
    scaled = [i for i, n in enumerate(names) if n.startswith(("num__", "log__"))]
    m["num_means_train"] = Xt[:, scaled].mean(axis=0).round(6).tolist()
    m["num_stds_train"] = Xt[:, scaled].std(axis=0).round(6).tolist()
    m["num_means_test"] = Xv[:, scaled].mean(axis=0).round(4).tolist()
    m["num_stds_test"] = Xv[:, scaled].std(axis=0).round(4).tolist()
    m["max_abs_z_train"] = np.abs(Xt[:, scaled]).max(axis=0).round(2).tolist()

    # A category never seen in training, and missing values everywhere, still give a finite row of the same width
    probe = X_test.iloc[:2].copy()
    probe.iloc[0, probe.columns.get_loc("gender")] = "Nonbinary"
    probe.iloc[0, probe.columns.get_loc("work_type")] = "Freelancer"
    probe.iloc[1] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # sklearn warns that the unseen categories become zeros
        out = pre.transform(probe)
    gender_cols = [i for i, n in enumerate(names) if n.startswith("cat__gender_")]
    work_cols = [i for i, n in enumerate(names) if n.startswith("cat__work_type_")]
    assert out.shape == (2, len(names)) and np.isfinite(out).all()
    assert out[0, gender_cols].sum() == 0 and out[0, work_cols].sum() == 0
    indicator = list(names).index("missing__missingindicator_bmi")
    m["unseen_category_probe"] = {"gender_block": out[0, gender_cols].tolist(),
                                  "work_type_block": out[0, work_cols].tolist(),
                                  "all_missing_row_finite": bool(np.isfinite(out[1]).all()),
                                  "all_missing_row_bmi_indicator": float(out[1, indicator])}


def neighborhood_stats(Z, y, profile):
    """Share of positives among the k nearest neighbors of each positive, and of identical categorical profiles."""
    nn = NearestNeighbors(n_neighbors=KNN).fit(Z)
    idx = nn.kneighbors(return_distance=False)   # no argument: each point is excluded from its own neighbors
    pos = y == 1
    enrichment = y[idx[pos]].mean() * 100
    agreement = (profile[idx] == profile[:, None]).mean() * 100
    return enrichment, agreement


def scatter_by_class(ax, Z, y, title, xlabel="Dimension 1 (no units)", ylabel="Dimension 2 (no units)"):
    """2D projection colored by the target, with the positives drawn larger and on top."""
    for k in (0, 1):
        sel = y == k
        ax.scatter(Z[sel, 0], Z[sel, 1], s=4 if k == 0 else 12, alpha=0.35 if k == 0 else 0.9,
                   color=CLASS_COLORS[k], label=f"{CLASS_LABELS[k]} · n = {sel.sum():,}", edgecolors="none",
                   rasterized=True)
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    ax.legend(loc="best", markerscale=2)


def main():
    """Run stage 4 and write Tables 13-18, Figures 9-13 and results/reduction_metrics.json."""
    res = Results("reduction")
    m = res.metrics
    raw = load_raw()
    X_train, X_test, y_train, y_test = split(raw)
    y = y_train.to_numpy()

    outliers(X_train, y_train, res)

    pre = build_preprocessor()
    Xt = pre.fit_transform(X_train)     # fit on train only
    Xv = pre.transform(X_test)          # test is only transformed
    names = pre.get_feature_names_out()
    pipeline_checks(pre, raw, X_train, X_test, Xt, Xv, names, res)

    # BMI missingness vs target in train (motivates the indicator)
    miss = X_train.bmi.isna().to_numpy()
    m["train_stroke_rate_bmi_missing"] = float(y[miss].mean() * 100)
    m["train_stroke_rate_bmi_observed"] = float(y[~miss].mean() * 100)
    m["train_bmi_missing_rate_pos"] = float(miss[y == 1].mean() * 100)
    m["train_bmi_missing_rate_neg"] = float(miss[y == 0].mean() * 100)

    # Figure 9 — what motivates the missing-value and the transform choices
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    ax = axes[0]
    for i, (label, sel) in enumerate([("bmi observed", ~miss), ("bmi missing", miss)]):
        k, n = int(y[sel].sum()), int(sel.sum())
        lo, hi = wilson_interval(k, n)
        ax.bar(i, k / n * 100, color=[CLASS_COLORS[0], CLASS_COLORS[1]][i], width=0.55,
               label=f"{label}: {k}/{n} = {k / n:.1%}")
        ax.errorbar(i, k / n * 100, yerr=[[k / n * 100 - lo * 100], [hi * 100 - k / n * 100]], color="#0b0b0b",
                    capsize=4)
    ax.axhline(y.mean() * 100, color=NEUTRAL, ls="--", label=f"Train base rate {y.mean():.2%}")
    ax.set_xticks([0, 1], ["bmi observed", "bmi missing"])
    ax.set(title="Stroke rate by bmi missingness (95% Wilson)", xlabel="bmi status", ylabel="Stroke rate (%)")
    ax.legend(loc="upper left")
    for ax, c in zip(axes[1:], SKEWED_FEATURES):
        raw = X_train[c].dropna()
        j = list(names).index(f"log__{c}")
        z = Xt[~miss, j] if c == "bmi" else Xt[:, j]
        ax.hist(z, bins=40, color=CLASS_COLORS[0], alpha=0.85, edgecolor="white", linewidth=0.4,
                label=f"log1p + standardized (n = {len(z):,})")
        ax.set(title=f"{c}: skewness {raw.skew():.2f} → {np.log1p(raw).skew():.2f}",
               xlabel=f"Standardized log1p({c}) (train z-score)", ylabel="Training rows")
        ax.axvline(0, color="#0b0b0b", lw=1, label="Train mean = 0")
        ax.legend(loc="upper right")
    save_figure(fig, 9, "What the pipeline answers: informative missingness and right skew")

    # PCA on the full transformed training matrix
    pca = PCA(random_state=SEED).fit(Xt)
    evr = pca.explained_variance_ratio_
    cum = np.cumsum(evr)
    P = pca.transform(Xt)
    m["pca_evr"] = evr.round(6).tolist()
    m["pca_pc1"], m["pca_pc2"], m["pca_pc12"] = float(evr[0] * 100), float(evr[1] * 100), float(cum[1] * 100)
    for t in (0.80, 0.90, 0.95):
        m[f"pca_k_{int(t * 100)}"] = int(np.searchsorted(cum, t) + 1)
    m["pca_zero_components"] = int((pca.explained_variance_ < 1e-10).sum())
    m["total_variance"] = float(pca.explained_variance_.sum())
    var_table = pd.DataFrame({"component": [f"PC{i + 1}" for i in range(8)], "explained (%)": evr[:8] * 100,
                              "cumulative (%)": cum[:8] * 100})
    res.table(16, "PCA explained variance (first 8 of 24 components)", var_table,
              {"explained (%)": ".2f", "cumulative (%)": ".2f"})
    load = pd.DataFrame({"feature": names, "PC1": pca.components_[0], "PC2": pca.components_[1]})
    top = load.reindex(load[["PC1", "PC2"]].abs().max(axis=1).sort_values(ascending=False).index).head(10)
    res.table(17, "Largest PCA loadings (eigenvector coefficients) of PC1 and PC2", top,
              {"PC1": "+.3f", "PC2": "+.3f"})
    m["loadings"] = load.set_index("feature").to_dict(orient="index")
    # correlation of the PC scores with the scaled numerical columns and the indicator (helps naming the components)
    for i in (0, 1):
        m[f"pc{i + 1}_corr"] = {n: float(np.corrcoef(P[:, i], Xt[:, j])[0, 1])
                                for j, n in enumerate(names) if not n.startswith("cat__")}

    # Figure 10 — PCA scatter, cumulative variance, loadings
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))
    scatter_by_class(axes[0], P, y, f"PC1 × PC2 ({cum[1]:.1%} of the variance)",
                     f"PC1 score ({evr[0]:.1%})", f"PC2 score ({evr[1]:.1%})")
    ax = axes[1]
    ax.bar(np.arange(1, len(evr) + 1), evr * 100, color=CLASS_COLORS[0], alpha=0.6, label="Explained by component")
    ax.plot(np.arange(1, len(evr) + 1), cum * 100, color=CLASS_COLORS[1], marker="o", ms=3, label="Cumulative")
    ax.axhline(90, color=NEUTRAL, ls="--", lw=1, label=f"90% reached at PC{m['pca_k_90']}")
    ax.set(title="Explained variance", xlabel="Principal component", ylabel="Variance explained (%)", ylim=(0, 103))
    ax.legend(loc="center right")
    ax = axes[2]
    show = top.iloc[::-1]
    pos = np.arange(len(show))
    ax.barh(pos - 0.2, show.PC1, height=0.4, color=CLASS_COLORS[0], label="PC1")
    ax.barh(pos + 0.2, show.PC2, height=0.4, color=CLASS_COLORS[1], label="PC2")
    ax.axvline(0, color="#0b0b0b", lw=0.8)
    ax.set_yticks(pos, show.feature, fontsize=8)
    ax.set(title="10 largest loadings", xlabel="Loading (eigenvector coefficient)", ylabel="Feature (pipeline output)")
    ax.legend(loc="lower right")
    save_figure(fig, 10, "PCA of the scaled training features: two components keep under half the variance")

    # t-SNE and UMAP, three parameter values each, on the same training matrix
    embeddings = {"PCA (2 components)": P[:, :2]}
    for perp in PERPLEXITIES:
        tsne = TSNE(n_components=2, perplexity=perp, init="pca", learning_rate="auto", random_state=SEED)
        embeddings[f"t-SNE perplexity {perp}"] = tsne.fit_transform(Xt)
        m[f"tsne_kl_{perp}"] = float(tsne.kl_divergence_)
    for nn in N_NEIGHBORS:
        um = UMAP(n_components=2, n_neighbors=nn, min_dist=0.1, random_state=SEED, n_jobs=1)
        embeddings[f"UMAP n_neighbors {nn}"] = um.fit_transform(Xt)

    # Null control (reduction handout): permute every raw column independently, which keeps each marginal
    # distribution and every categorical profile valid but destroys the associations between columns and target
    rng = np.random.default_rng(SEED)
    shuffled = X_train.apply(lambda col: col.to_numpy()[rng.permutation(len(col))])
    Xs = pre.transform(shuffled)
    ys = y[rng.permutation(len(y))]
    tsne = TSNE(n_components=2, perplexity=30, init="pca", learning_rate="auto", random_state=SEED)
    Zs = tsne.fit_transform(Xs)

    # Diagnostics: neighborhood preservation, class enrichment and categorical-profile agreement
    cat_cols = [i for i, n in enumerate(names) if n.startswith("cat__") or n.startswith("missing__")]
    profile = np.array(["".join(str(int(v)) for v in row) for row in Xt[:, cat_cols]])
    m["n_profiles"] = int(len(set(profile)))
    rows = []
    enr, agr = neighborhood_stats(Xt, y, profile)
    rows.append({"space": "Original 24 dimensions", "trustworthiness k=5": "—", "trustworthiness k=30": "—",
                 "positives among 10-NN of positives (%)": enr, "same categorical profile among 10-NN (%)": agr})
    for name, Z in embeddings.items():
        enr, agr = neighborhood_stats(Z, y, profile)
        rows.append({"space": name, "trustworthiness k=5": trustworthiness(Xt, Z, n_neighbors=5),
                     "trustworthiness k=30": trustworthiness(Xt, Z, n_neighbors=30),
                     "positives among 10-NN of positives (%)": enr, "same categorical profile among 10-NN (%)": agr})
    cat_s = np.array(["".join(str(int(v)) for v in row) for row in Xs[:, cat_cols]])
    enr, agr = neighborhood_stats(Zs, ys, cat_s)
    rows.append({"space": "Control: t-SNE perplexity 30 on independently shuffled columns",
                 "trustworthiness k=5": trustworthiness(Xs, Zs, n_neighbors=5),
                 "trustworthiness k=30": trustworthiness(Xs, Zs, n_neighbors=30),
                 "positives among 10-NN of positives (%)": enr, "same categorical profile among 10-NN (%)": agr})
    m["n_profiles_shuffled"] = int(len(set(cat_s)))
    diag = pd.DataFrame(rows)
    res.table(18, f"Projection diagnostics (train, {len(y):,} rows; base rate {y.mean():.2%})", diag,
              {"trustworthiness k=5": ".3f", "trustworthiness k=30": ".3f",
               "positives among 10-NN of positives (%)": ".2f", "same categorical profile among 10-NN (%)": ".1f"})
    m["projection_diagnostics"] = diag.to_dict(orient="records")

    for number, method, params, key in [(11, "t-SNE", PERPLEXITIES, "perplexity"),
                                        (12, "UMAP", N_NEIGHBORS, "n_neighbors")]:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5.4))
        for ax, p in zip(axes, params):
            scatter_by_class(ax, embeddings[f"{method} {key} {p}"], y, f"{key} = {p}")
        save_figure(fig, number, f"{method} of the scaled training features: three {key} values, colored by the target")

    # Figure 13 — what the nonlinear maps group by: color by a categorical feature and by age
    fig, axes = plt.subplots(3, 2, figsize=(12, 16))   # one row per method: work_type on the left, age on the right
    shown = ["PCA (2 components)", "t-SNE perplexity 30", "UMAP n_neighbors 15"]
    work = X_train.work_type.to_numpy()
    order = ["Private", "Self-employed", "Govt_job", "children", "Never_worked"]
    axis_names = {"PCA (2 components)": ("PC1 score", "PC2 score")}
    for ax, name in zip(axes[:, 0], shown):
        Z = embeddings[name]
        for k, color in zip(order, GROUP_COLORS):
            sel = work == k
            ax.scatter(Z[sel, 0], Z[sel, 1], s=4, alpha=0.6, color=color, label=f"{k} (n={sel.sum():,})",
                       edgecolors="none", rasterized=True)
        xl, yl = axis_names.get(name, ("Dimension 1 (no units)", "Dimension 2 (no units)"))
        ax.set(title=f"{name} — colored by work_type", xlabel=xl, ylabel=yl)
        ax.legend(loc="best", markerscale=3, fontsize=7)
    for ax, name in zip(axes[:, 1], shown):
        Z = embeddings[name]
        sc = ax.scatter(Z[:, 0], Z[:, 1], c=X_train.age, cmap="Blues", s=4, edgecolors="none", rasterized=True)
        pos = y == 1
        ax.scatter(Z[pos, 0], Z[pos, 1], s=14, facecolors="none", edgecolors=CLASS_COLORS[1], lw=0.8,
                   label=f"{CLASS_LABELS[1]} · n = {pos.sum()}")
        fig.colorbar(sc, ax=ax, fraction=0.046, label=UNITS["age"])
        xl, yl = axis_names.get(name, ("Dimension 1 (no units)", "Dimension 2 (no units)"))
        ax.set(title=f"{name} — colored by age", xlabel=xl, ylabel=yl)
        ax.legend(loc="best", fontsize=7)
    save_figure(fig, 13, "What the maps group by: categorical profiles make the islands, age orders the positives")

    m["versions"] = {p: importlib.metadata.version(p) for p in
                     ["numpy", "pandas", "scipy", "scikit-learn", "umap-learn", "matplotlib"]}
    m["python"] = platform.python_version()
    res.write()
    print(f"Stage 4 done: train {Xt.shape}, test {Xv.shape}, Figures 9-13 written.")


if __name__ == "__main__":
    main()

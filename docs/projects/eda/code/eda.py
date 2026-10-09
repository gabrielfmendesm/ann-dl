"""Stages 1-3 of the EDA: inspection, univariate and bivariate analysis (Figures 1-8).

The full file is read only for the stage-1 inventory (missing values, duplicates,
consistency checks and the target distribution). The split happens right after it,
and every later statistic and figure uses the training rows only.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from scipy.stats import chi2_contingency, fisher_exact, mannwhitneyu, spearmanr

from common import (CLASS_COLORS, CLASS_LABELS, GROUP_COLORS, NEUTRAL, RESULTS_DIR, UNITS, Results,
                    save_figure, wilson_interval)
from preprocessing import (CATEGORICAL_FEATURES, FEATURES, ID_COLUMN, NUMERICAL_FEATURES, SEED, TARGET,
                           load_raw, split)

NUM = NUMERICAL_FEATURES
CAT = CATEGORICAL_FEATURES
RARE_SHARE = 0.01          # a category below 1% of the training rows is flagged as rare
REDUNDANT_RHO = 0.80       # |rho| at or above this would flag a redundant numerical pair
VOCABULARY = {
    "gender": {"Female", "Male", "Other"}, "hypertension": {0, 1}, "heart_disease": {0, 1},
    "ever_married": {"Yes", "No"}, "work_type": {"Private", "Self-employed", "Govt_job", "children", "Never_worked"},
    "Residence_type": {"Urban", "Rural"}, "smoking_status": {"formerly smoked", "never smoked", "smokes", "Unknown"},
    "stroke": {0, 1},
}


def stage1(raw, res):
    """1B quality, 1C target and 1D split; returns the training frame (features + target)."""
    m = res.metrics
    m["raw_shape"] = list(raw.shape)

    # 1B — missing values per column, full file (the inventory precedes the split)
    missing = pd.DataFrame({"column": raw.columns, "missing (count)": raw.isna().sum().values,
                            "missing (%)": raw.isna().mean().values * 100})
    res.table(2, "Missing values per column (full file)", missing, {"missing (%)": ".2f"})
    m["bmi_missing"] = int(raw["bmi"].isna().sum())
    m["bmi_missing_pct"] = float(raw["bmi"].isna().mean() * 100)

    # 1B — duplicates, impossible and inconsistent values
    checks = {
        "Fully duplicated rows": raw.duplicated().sum(),
        "Duplicated rows ignoring id": raw.drop(columns=ID_COLUMN).duplicated().sum(),
        "Repeated id": raw[ID_COLUMN].duplicated().sum(),
        "Constant columns": sum(raw[c].nunique(dropna=False) == 1 for c in raw.columns),
        "Values outside the documented vocabulary": sum((~raw[c].isin(v)).sum() for c, v in VOCABULARY.items()),
        "age <= 0 or age > 120": ((raw.age <= 0) | (raw.age > 120)).sum(),
        "Fractional age (all below 2 years)": (raw.age % 1 != 0).sum(),
        "avg_glucose_level <= 0": (raw.avg_glucose_level <= 0).sum(),
        "bmi < 12 (observed)": (raw.bmi < 12).sum(),
        "bmi > 60 (observed)": (raw.bmi > 60).sum(),
        "work_type = children with age >= 18": ((raw.work_type == "children") & (raw.age >= 18)).sum(),
        "ever_married = Yes with age < 18": ((raw.ever_married == "Yes") & (raw.age < 18)).sum(),
        "smoking_status = smokes with age < 12": ((raw.smoking_status == "smokes") & (raw.age < 12)).sum(),
        "smoking_status = Unknown": (raw.smoking_status == "Unknown").sum(),
        "smoking_status = Unknown with age < 18": ((raw.smoking_status == "Unknown") & (raw.age < 18)).sum(),
    }
    checks = {k: int(v) for k, v in checks.items()}
    m["checks"] = checks
    res.table(3, "Duplicates, impossible and inconsistent values (full file)",
              pd.DataFrame({"check": list(checks), "rows": list(checks.values())}))
    m["bmi_extremes"] = sorted(raw.bmi.dropna().nlargest(4).tolist(), reverse=True)
    m["age_min"], m["age_max"] = float(raw.age.min()), float(raw.age.max())
    m["gender_other_rows"] = int((raw.gender == "Other").sum())
    m["id_distinct"] = int(raw[ID_COLUMN].nunique())
    m["fractional_age_max"] = float(raw.age[raw.age % 1 != 0].max())
    m["children_age_max"] = float(raw.age[raw.work_type == "children"].max())
    m["never_worked_age_range"] = [float(raw.age[raw.work_type == "Never_worked"].min()),
                                   float(raw.age[raw.work_type == "Never_worked"].max())]
    m["smokers_under_12_ages"] = raw.age[(raw.smoking_status == "smokes") & (raw.age < 12)].tolist()
    m["bmi_below_12"] = raw.loc[raw.bmi < 12, ["age", "bmi"]].values.tolist()

    # 1B — leakage audit: the identifier, the row order and the bmi missingness pattern
    rho_id, p_id = spearmanr(raw[ID_COLUMN], raw[TARGET])
    positives_at = np.flatnonzero(raw[TARGET].to_numpy() == 1)
    m["id_target_spearman"], m["id_target_p"] = float(rho_id), float(p_id)
    m["positives_first_row"], m["positives_last_row"] = int(positives_at.min()), int(positives_at.max())
    m["last20_positives"] = int(raw[TARGET].iloc[int(len(raw) * 0.8):].sum())
    by_missing = raw.groupby(raw.bmi.isna())[TARGET].agg(["count", "sum", "mean"])
    m["stroke_rate_bmi_missing_full"] = float(by_missing.loc[True, "mean"] * 100)
    m["stroke_rate_bmi_observed_full"] = float(by_missing.loc[False, "mean"] * 100)
    m["strokes_bmi_missing_full"] = int(by_missing.loc[True, "sum"])

    # 1C — target distribution
    counts = raw[TARGET].value_counts().sort_index()
    m["n_pos"], m["n_neg"] = int(counts[1]), int(counts[0])
    m["pos_pct"] = float(counts[1] / len(raw) * 100)
    m["imbalance_ratio"] = float(counts[0] / counts[1])

    # 1D — stratified split before any preprocessing decision
    X_train, X_test, y_train, y_test = split(raw)
    assert set(X_train.index).isdisjoint(X_test.index) and len(X_train) + len(X_test) == len(raw)
    m["train_rows"], m["test_rows"] = len(X_train), len(X_test)
    pd.concat([pd.DataFrame({"row": X_train.index, "id": raw.loc[X_train.index, ID_COLUMN], "partition": "train"}),
               pd.DataFrame({"row": X_test.index, "id": raw.loc[X_test.index, ID_COLUMN], "partition": "test"})]) \
        .to_csv(RESULTS_DIR / "split.csv", index=False)
    target_rows = []
    for part, y in [("Full file", raw[TARGET]), ("Train", y_train), ("Test", y_test)]:
        target_rows.append({"partition": part, "rows": len(y), "stroke = 0": int((y == 0).sum()),
                            "stroke = 1": int((y == 1).sum()), "stroke = 1 (%)": float(y.mean() * 100)})
    target_table = pd.DataFrame(target_rows)
    res.table(4, "Target distribution in the full file and after the stratified split", target_table,
              {"stroke = 1 (%)": ".2f"})
    m["train_pos"], m["test_pos"] = int(y_train.sum()), int(y_test.sum())
    m["train_pos_pct"], m["test_pos_pct"] = float(y_train.mean() * 100), float(y_test.mean() * 100)
    m["train_bmi_missing"] = int(X_train.bmi.isna().sum())
    m["train_bmi_missing_pct"] = float(X_train.bmi.isna().mean() * 100)

    # Figure 1 — class counts and where the positives sit in the file
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    ax = axes[0]
    bars = ax.bar([CLASS_LABELS[0], CLASS_LABELS[1]], counts.values, color=[CLASS_COLORS[0], CLASS_COLORS[1]],
                  width=0.6)
    for bar, n in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, n + 60, f"{n:,} ({n / len(raw):.2%})", ha="center")
    ax.set(title=f"Class frequencies (full file, {len(raw):,} rows)", xlabel="Target class", ylabel="Rows",
           ylim=(0, counts.max() * 1.15))
    ax.legend(handles=[Patch(color=CLASS_COLORS[k], label=CLASS_LABELS[k]) for k in (0, 1)], loc="upper right")
    ax = axes[1]
    row = np.arange(len(raw))
    ax.plot(row, raw[TARGET].cumsum(), color=CLASS_COLORS[1], lw=2, label="Cumulative stroke = 1 rows")
    ax.axvline(int(len(raw) * 0.8), color=NEUTRAL, ls="--", lw=1.2, label="Last 20% of the file (unshuffled 'test')")
    ax.set(title="Positives by position in the CSV: the file is sorted by target",
           xlabel="Row position in the CSV", ylabel="Cumulative stroke = 1 rows")
    ax.legend(loc="lower right")
    save_figure(fig, 1, f"Target: {m['pos_pct']:.2f}% positives, all stored at the top of the file")

    train = X_train.assign(**{TARGET: y_train})
    return train


def stage2(train, res):
    """2A numerical and 2B categorical distributions (training rows only)."""
    m = res.metrics
    rows = []
    for c in NUM:
        x = train[c].dropna()
        rows.append({"feature": c, "count": len(x), "missing": int(train[c].isna().sum()),
                     "mean": x.mean(), "median": x.median(), "std": x.std(), "min": x.min(),
                     "Q1": x.quantile(.25), "Q3": x.quantile(.75), "max": x.max(),
                     "skewness": x.skew(), "excess kurtosis": x.kurt()})
    stats = pd.DataFrame(rows)
    res.table(5, "Descriptive statistics of the numerical features (train, observed values)", stats,
              {k: ".2f" for k in ["mean", "median", "std", "min", "Q1", "Q3", "max", "skewness", "excess kurtosis"]})
    m["numerical_stats"] = stats.set_index("feature").to_dict(orient="index")

    # Glucose: locate the two modes and the valley between them on a fixed histogram
    glucose = train.avg_glucose_level
    hist, edges = np.histogram(glucose, bins=np.arange(50, 280, 10))
    centers = (edges[:-1] + edges[1:]) / 2
    low = centers < 150
    valley_zone = (centers > 130) & (centers < 190)
    m["glucose_mode_low"] = float(centers[low][np.argmax(hist[low])])
    m["glucose_mode_high"] = float(centers[~low][np.argmax(hist[~low])])
    m["glucose_valley"] = float(centers[valley_zone][np.argmin(hist[valley_zone])])
    m["glucose_above_150"] = int((glucose > 150).sum())
    m["glucose_above_150_pct"] = float((glucose > 150).mean() * 100)

    # Figure 2 — histograms with mean/median and boxplots (shape, modes, outliers)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), gridspec_kw={"height_ratios": [3, 1.5]})
    for j, c in enumerate(NUM):
        x = train[c].dropna()
        ax = axes[0, j]
        bins = np.arange(0, 84, 2) if c == "age" else 40   # 2-year bins aligned on integer ages
        ax.hist(x, bins=bins, color=CLASS_COLORS[0], alpha=0.85, edgecolor="white", linewidth=0.4,
                label=f"Training rows (n = {len(x):,})")
        ax.axvline(x.mean(), color="#0b0b0b", lw=1.5, label=f"Mean = {x.mean():.1f}")
        ax.axvline(x.median(), color=CLASS_COLORS[1], lw=1.5, ls="--", label=f"Median = {x.median():.1f}")
        ax.set(title=f"{c} — skewness {x.skew():.2f}", xlabel=UNITS[c], ylabel="Rows")
        ax.legend(loc="upper left" if c == "age" else "upper right")
        ax = axes[1, j]
        ax.boxplot(x, orientation="horizontal", widths=0.45, patch_artist=True,
                   boxprops={"facecolor": "#b7d3f6"}, medianprops={"color": CLASS_COLORS[1]},
                   flierprops={"marker": "o", "markersize": 2.5, "alpha": 0.4})
        q1, q3 = x.quantile(.25), x.quantile(.75)
        n_out = int(((x < q1 - 1.5 * (q3 - q1)) | (x > q3 + 1.5 * (q3 - q1))).sum())
        ax.set(xlabel=UNITS[c], yticks=[], ylabel="Train", ylim=(0.5, 1.7),
               title=f"Boxplot: {n_out} points beyond 1.5×IQR")
        ax.legend(handles=[Patch(facecolor="#b7d3f6", edgecolor="#0b0b0b",
                                 label="Box: Q1–Q3 · orange line: median · whiskers: 1.5×IQR"),
                           Line2D([], [], marker="o", color="#0b0b0b", ls="none", ms=3, alpha=0.4,
                                  label=f"{n_out} points beyond the fences")],
                  loc="upper right", fontsize=7)
    axes[0, 1].annotate("second mode", xy=(m["glucose_mode_high"], 40), xytext=(225, 160),
                        arrowprops={"arrowstyle": "->"}, ha="center")
    save_figure(fig, 2, "Numerical features (train): near-symmetric age, bimodal glucose, right-skewed BMI")

    # 2B — frequencies, cardinality and rare categories
    rows = []
    for c in CAT:
        vc = train[c].value_counts()
        for k, n in vc.items():
            rows.append({"feature": c, "category": str(k), "count": int(n), "share (%)": n / len(train) * 100,
                         "cardinality": len(vc), "rare (< 1%)": "yes" if n / len(train) < RARE_SHARE else ""})
    freq = pd.DataFrame(rows)
    res.table(6, "Frequencies and cardinality of the categorical features (train)", freq, {"share (%)": ".2f"})
    m["rare_categories"] = freq.loc[freq["rare (< 1%)"] == "yes", ["feature", "category", "count", "share (%)"]] \
        .to_dict(orient="records")
    m["smoking_unknown_train"] = int((train.smoking_status == "Unknown").sum())
    m["smoking_unknown_train_pct"] = float((train.smoking_status == "Unknown").mean() * 100)
    m["smoking_unknown_under18_train"] = int(((train.smoking_status == "Unknown") & (train.age < 18)).sum())
    m["children_rows_train"] = int((train.work_type == "children").sum())
    m["children_unknown_smoking_train"] = int(((train.work_type == "children")
                                               & (train.smoking_status == "Unknown")).sum())

    # Figure 3 — one bar panel per categorical feature + cardinality panel
    fig, axes = plt.subplots(4, 2, figsize=(12, 15))
    for ax, c in zip(axes.flat, CAT):
        vc = train[c].value_counts().sort_values()
        shares = vc / len(train) * 100
        colors = [CLASS_COLORS[1] if s < RARE_SHARE * 100 else CLASS_COLORS[0] for s in shares]
        ax.barh([str(k) for k in vc.index], vc.values, color=colors, height=0.6)
        for i, (n, s) in enumerate(zip(vc.values, shares)):
            ax.text(n + vc.max() * 0.02, i, f"{n:,} ({s:.1f}%)" if s >= 1 else f"{n:,} ({s:.2f}%)", va="center",
                    fontsize=8)
        ax.set(title=f"{c} — {len(vc)} categories", xlabel="Training rows", ylabel="Category",
               xlim=(0, vc.max() * 1.35))
    ax = axes.flat[-1]
    card = pd.Series({c: train[c].nunique() for c in CAT}).sort_values()
    ax.barh(card.index, card.values, color=NEUTRAL, height=0.6, label="Distinct categories")
    ax.set(title=f"Cardinality (id: {m['id_distinct']:,} distinct, dropped)", xlabel="Distinct categories in train",
           ylabel="Feature", xticks=range(0, 7))
    ax.legend(loc="lower right")
    fig.legend(handles=[Patch(color=CLASS_COLORS[0], label="Category with ≥ 1% of the training rows"),
                        Patch(color=CLASS_COLORS[1], label="Rare category: < 1% of the training rows")],
               loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.015), fontsize=10)
    save_figure(fig, 3, f"Categorical features (train): low cardinality, {len(m['rare_categories'])} rare categories")


def stage3(train, res):
    """3A numerical x numerical, 3B categorical x target, 3C numerical x categorical (train only)."""
    m = res.metrics
    y = train[TARGET]

    # 3A — Pearson and Spearman on pairwise-complete observed values
    rows = []
    for i, a in enumerate(NUM):
        for b in NUM[i + 1:]:
            pair = train[[a, b]].dropna()
            rows.append({"pair": f"{a} × {b}", "n": len(pair), "Pearson r": pair[a].corr(pair[b]),
                         "Spearman ρ": spearmanr(pair[a], pair[b]).statistic})
    corr = pd.DataFrame(rows)
    res.table(7, "Pairwise correlations of the numerical features (train, observed values)", corr,
              {"Pearson r": ".3f", "Spearman ρ": ".3f"})
    top = corr.loc[corr["Spearman ρ"].abs().idxmax()]
    m["top_pair"], m["top_pair_rho"], m["top_pair_r"], m["top_pair_n"] = (
        top["pair"], float(top["Spearman ρ"]), float(top["Pearson r"]), int(top["n"]))
    m["redundant_pairs"] = int((corr["Spearman ρ"].abs() >= REDUNDANT_RHO).sum())
    adults = train[train.age >= 18][["age", "bmi"]].dropna()
    m["age_bmi_rho_adults"] = float(spearmanr(adults.age, adults.bmi).statistic)
    m["age_bmi_n_adults"] = len(adults)

    # Figure 4 — the two coefficients side by side (method justification)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, method in zip(axes, ["pearson", "spearman"]):
        mat = train[NUM].corr(method=method)
        im = ax.imshow(mat, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(3), NUM, rotation=20)
        ax.set_yticks(range(3), NUM)
        ax.grid(False)
        for r in range(3):
            for c in range(3):
                ax.text(c, r, f"{mat.iloc[r, c]:.3f}", ha="center", va="center",
                        color="white" if abs(mat.iloc[r, c]) > 0.6 else "#0b0b0b")
        ax.set(title=f"{method.capitalize()} (pairwise-complete)", xlabel="Feature", ylabel="Feature")
        fig.colorbar(im, ax=ax, fraction=0.046, label=f"{method.capitalize()} coefficient")
        i, j = (NUM.index(f) for f in top["pair"].split(" × "))
        for r, c in ((i, j), (j, i)):
            ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False, edgecolor="#0b0b0b", lw=2))
    fig.legend(handles=[Patch(facecolor="none", edgecolor="#0b0b0b", lw=2,
                              label=f"Strongest pair: {top['pair']} — no pair reaches the redundancy flag "
                                    f"|ρ| ≥ {REDUNDANT_RHO:.2f}")],
               loc="lower center", bbox_to_anchor=(0.5, -0.04), fontsize=9)
    save_figure(fig, 4, "Numerical correlations (train): no redundant pair, strongest is age × BMI")

    # Figure 5 — scatter plots of the three pairs, positives drawn on top
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for ax, (_, r) in zip(axes, corr.iterrows()):
        a, b = r["pair"].split(" × ")
        for k in (0, 1):
            sel = train[y == k]
            ax.scatter(sel[a], sel[b], s=6 if k == 0 else 14, alpha=0.3 if k == 0 else 0.85,
                       color=CLASS_COLORS[k], label=CLASS_LABELS[k], edgecolors="none")
        ax.set(title=f"ρ = {r['Spearman ρ']:.3f} · r = {r['Pearson r']:.3f} · n = {r['n']:,}",
               xlabel=UNITS[a], ylabel=UNITS[b])
        ax.legend(loc="upper left", markerscale=1.5)
    save_figure(fig, 5, "Numerical pairs (train): broad clouds, stroke cases concentrated at older ages")

    # 3B — stroke rate per category with 95% Wilson intervals; chi-square and Cramér's V per feature
    base = y.mean() * 100
    m["train_base_rate"] = float(base)
    rows, assoc = [], []
    for c in CAT:
        g = train.groupby(c)[TARGET].agg(["count", "sum"])
        for k, r in g.iterrows():
            lo, hi = wilson_interval(r["sum"], r["count"])
            rows.append({"feature": c, "category": str(k), "n": int(r["count"]), "strokes": int(r["sum"]),
                         "stroke rate (%)": r["sum"] / r["count"] * 100, "ci_lo": lo * 100, "ci_hi": hi * 100})
        table = pd.crosstab(train[c], y)
        # Pearson's chi-square without Yates' continuity correction (the correction only applies to 2x2 tables)
        chi2, p, dof, expected = chi2_contingency(table, correction=False)
        # expected counts below 5 invalidate the approximation: drop those categories and re-test
        small = table.index[(expected < 5).any(axis=1)]
        if len(small):
            kept = table.drop(index=small)
            chi2_k, p_k, _, _ = chi2_contingency(kept, correction=False)
            p_kept = f"{p_k:.2g} (without {', '.join(map(str, small))})"
        else:
            p_kept = "—"
        assoc.append({"feature": c, "χ²": chi2, "dof": dof, "p-value": p,
                      "Cramér's V": np.sqrt(chi2 / (table.values.sum() * (min(table.shape) - 1))),
                      "min expected count": expected.min(), "p-value without the sparse categories": p_kept})
    rates = pd.DataFrame(rows)
    rates_table = rates.assign(**{"95% CI (%)": [f"{lo:.1f}–{hi:.1f}" for lo, hi in zip(rates.ci_lo, rates.ci_hi)]}) \
        .drop(columns=["ci_lo", "ci_hi"])
    res.table(8, "Stroke rate per category (train) with 95% Wilson intervals", rates_table, {"stroke rate (%)": ".2f"})
    assoc = pd.DataFrame(assoc).sort_values("Cramér's V", ascending=False)
    res.table(9, "Association of each categorical feature with the target (train, χ² test of independence)", assoc,
              {"χ²": ".1f", "p-value": ".2g", "Cramér's V": ".3f", "min expected count": ".2f"})
    m["category_rates"] = rates_table.to_dict(orient="records")
    m["cramers_v"] = assoc.set_index("feature")["Cramér's V"].to_dict()
    m["chi2_p"] = assoc.set_index("feature")["p-value"].to_dict()

    # Figure 6 — category stroke rates with intervals, plus the Cramér's V ranking
    fig, axes = plt.subplots(4, 2, figsize=(12, 16))
    for ax, c in zip(axes.flat, CAT):
        sub = rates[rates.feature == c].reset_index(drop=True)
        rate, lo, hi = (sub[k].to_numpy() for k in ("stroke rate (%)", "ci_lo", "ci_hi"))
        ypos = np.arange(len(sub))
        ax.barh(ypos, rate, color=CLASS_COLORS[1], height=0.55, label="Stroke rate in category")
        ax.errorbar(rate, ypos, xerr=[rate - lo, hi - rate], fmt="none", ecolor="#0b0b0b",
                    capsize=3, lw=1, label="95% Wilson interval")
        ax.axvline(base, color=CLASS_COLORS[0], ls="--", lw=1.3, label=f"Train base rate {base:.2f}%")
        ax.set_yticks(ypos, [f"{k} (n={n:,})" for k, n in zip(sub.category, sub.n)])
        ax.set(title=c, xlabel="Stroke rate (%)", ylabel="Category (rows)", xlim=(0, 25))
        for i, upper in enumerate(hi):
            if upper > 25:  # tiny groups: the interval runs off the shared axis, so state its upper end
                ax.text(24.5, i + 0.3, f"interval to {upper:.0f}%", ha="right", fontsize=7)
        ax.legend(loc="lower right", fontsize=7)
    ax = axes.flat[-1]
    ax.barh(assoc.feature[::-1], assoc["Cramér's V"][::-1], color=NEUTRAL, height=0.55, label="Cramér's V")
    ax.set(title="Strength of association with stroke", xlabel="Cramér's V (0 = none)", ylabel="Feature")
    ax.legend(loc="lower right")
    binary = ["hypertension", "heart_disease", "ever_married"]
    ratios = {c: float(rates.loc[rates.feature == c, "stroke rate (%)"].max()
                       / rates.loc[rates.feature == c, "stroke rate (%)"].min()) for c in binary}
    m["binary_rate_ratios"] = ratios
    save_figure(fig, 6, "Stroke rate by category (train): hypertension, heart disease and marriage multiply it by "
                        f"{min(ratios.values()):.1f}–{max(ratios.values()):.1f}")

    # 3C — grouped location (median) and spread (IQR) summaries
    def grouped(by, features):
        """Observed count, median, quartiles and IQR of each feature within each level of `by`."""
        out = []
        for f in features:
            for k, g in train.groupby(by):
                x = g[f].dropna()
                out.append({"feature": f, "group": f"{by} = {k}", "n": len(x), "median": x.median(),
                            "Q1": x.quantile(.25), "Q3": x.quantile(.75), "IQR": x.quantile(.75) - x.quantile(.25)})
        return pd.DataFrame(out)

    by_target = grouped(TARGET, NUM)
    # effect size next to the p-value: the single-feature AUC = U / (n1 · n0) measures class overlap
    pos = {f: train.loc[y == 1, f].dropna() for f in NUM}
    neg = {f: train.loc[y == 0, f].dropna() for f in NUM}
    tests = {f: mannwhitneyu(pos[f], neg[f]) for f in NUM}
    by_target["Mann–Whitney p"] = by_target.feature.map({f: t.pvalue for f, t in tests.items()})
    by_target["AUC (feature alone)"] = by_target.feature.map(
        {f: t.statistic / (len(pos[f]) * len(neg[f])) for f, t in tests.items()})
    res.table(10, "Numerical features by target class (train): location and spread", by_target,
              {"median": ".2f", "Q1": ".2f", "Q3": ".2f", "IQR": ".2f", "Mann–Whitney p": ".2g",
               "AUC (feature alone)": ".3f"})
    m["by_target"] = by_target.to_dict(orient="records")
    m["positives_age_ge_60_pct"] = float((train.loc[y == 1, "age"] >= 60).mean() * 100)
    m["glucose_above_150_stroke_rate"] = float(train.loc[train.avg_glucose_level > 150, TARGET].mean() * 100)
    m["glucose_below_150_stroke_rate"] = float(train.loc[train.avg_glucose_level <= 150, TARGET].mean() * 100)

    # Figure 7 — numerical features by class
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for ax, f in zip(axes, NUM):
        data = [train.loc[y == k, f].dropna() for k in (0, 1)]
        bp = ax.boxplot(data, widths=0.55, patch_artist=True, medianprops={"color": "#0b0b0b"},
                        flierprops={"marker": "o", "markersize": 2.5, "alpha": 0.35})
        for patch, k in zip(bp["boxes"], (0, 1)):
            patch.set_facecolor(CLASS_COLORS[k])
            patch.set_alpha(0.75)
        ax.set_xticks([1, 2], [f"{CLASS_LABELS[k]}\nn = {len(d):,}" for k, d in zip((0, 1), data)])
        iqr = [d.quantile(.75) - d.quantile(.25) for d in data]
        ax.set(title=f"{f}: median {data[0].median():.1f} → {data[1].median():.1f}, "
                     f"IQR {iqr[0]:.1f} → {iqr[1]:.1f}", xlabel="Target class", ylabel=UNITS[f])
    fig.legend(handles=[Patch(color=CLASS_COLORS[k], alpha=0.75, label=CLASS_LABELS[k]) for k in (0, 1)],
               loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.06), fontsize=10)
    save_figure(fig, 7, "Numerical features by class (train): positives are older, with higher glucose")

    # Figure 8 — confounding structure: age by work type and by smoking status, glucose by hypertension
    groups = [("work_type", "age"), ("smoking_status", "age"), ("hypertension", "avg_glucose_level")]
    other = pd.concat([grouped(by, [f]) for by, f in groups], ignore_index=True)
    res.table(11, "Numerical features grouped by categorical features (train): location and spread", other,
              {"median": ".2f", "Q1": ".2f", "Q3": ".2f", "IQR": ".2f"})
    m["grouped_other"] = other.to_dict(orient="records")
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
    for ax, (by, f) in zip(axes, groups):
        order = train.groupby(by)[f].median().sort_values().index.tolist()
        data = [train.loc[train[by] == k, f].dropna() for k in order]
        bp = ax.boxplot(data, widths=0.55, patch_artist=True, medianprops={"color": "#0b0b0b"},
                        flierprops={"marker": "o", "markersize": 2.5, "alpha": 0.35})
        for patch, color in zip(bp["boxes"], GROUP_COLORS):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_xticks(range(1, len(order) + 1), [f"{k}\nn={len(d):,}" for k, d in zip(order, data)], fontsize=8)
        ax.set(title=f"{f} by {by}", xlabel=by, ylabel=UNITS[f])
        ax.legend(handles=[Patch(color=c, alpha=0.7, label=f"{by} = {k}") for k, c in zip(order, GROUP_COLORS)],
                  loc="upper left", fontsize=7)
    save_figure(fig, 8, "Grouped boxplots (train): work type and smoking status are age proxies")

    # Age-stratified check of the 3B associations: does the gap survive within an age band?
    old_age = train.age >= 60
    adult = train.age >= 18
    rows = []
    for c in ["hypertension", "heart_disease", "ever_married", "work_type", "smoking_status"]:
        for k in sorted(train[c].unique(), key=str):
            sel = train[c] == k
            if not (sel & old_age).any():
                continue  # a category with no member aged 60 or more has no rate in that band
            rows.append({"feature": c, "category": str(k), "rate, all ages (%)": y[sel].mean() * 100,
                         "n, age ≥ 18": int((sel & adult).sum()),
                         "rate, age ≥ 18 (%)": y[sel & adult].mean() * 100,
                         "n, age ≥ 60": int((sel & old_age).sum()),
                         "rate, age ≥ 60 (%)": y[sel & old_age].mean() * 100})
    strat = pd.DataFrame(rows)
    # Fisher's exact test needs a 2x2 table, so only the binary features are tested within the 60+ band
    fisher = {}
    for c in ["hypertension", "heart_disease", "ever_married"]:
        sub = train[old_age]
        fisher[c] = float(fisher_exact(pd.crosstab(sub[c], sub[TARGET])).pvalue)
    strat["Fisher p, age ≥ 60"] = strat.feature.map(lambda f: f"{fisher[f]:.2g}" if f in fisher else "—")
    m["fisher_60"] = fisher
    nm = train[old_age & (train.ever_married == "No")][TARGET]
    m["never_married_60_strokes"], m["never_married_60_n"] = int(nm.sum()), len(nm)
    m["never_married_60_wilson"] = [x * 100 for x in wilson_interval(nm.sum(), len(nm))]
    mm = train[old_age & (train.ever_married == "Yes")][TARGET]
    m["married_60_wilson"] = [x * 100 for x in wilson_interval(mm.sum(), len(mm))]
    res.table(12, "Stroke rate per category within age bands (train): what remains after holding age roughly fixed",
              strat, {"rate, all ages (%)": ".2f", "rate, age ≥ 18 (%)": ".2f", "rate, age ≥ 60 (%)": ".2f"})
    m["age_stratified"] = strat.to_dict(orient="records")
    m["rate_age_ge_60"], m["rate_age_lt_60"] = float(y[old_age].mean() * 100), float(y[~old_age].mean() * 100)
    m["positives_age_ge_60"] = int(y[old_age].sum())
    m["age_median_by_married"] = train.groupby("ever_married").age.median().to_dict()
    high = train.avg_glucose_level > 150
    m["glucose_above_150_by_hypertension"] = train.groupby("hypertension").avg_glucose_level \
        .apply(lambda x: float((x > 150).mean() * 100)).to_dict()
    m["glucose_above_150_age_median"] = float(train.loc[high, "age"].median())
    m["glucose_below_150_age_median"] = float(train.loc[~high, "age"].median())


def main():
    """Run stages 1-3 and write Tables 2-12, Figures 1-8 and results/eda_metrics.json."""
    res = Results("eda")
    raw = load_raw()
    assert raw.shape == (5110, 12) and raw[ID_COLUMN].is_unique
    res.metrics["seed"] = SEED
    res.metrics["features"] = FEATURES
    train = stage1(raw, res)
    stage2(train, res)
    stage3(train, res)
    res.write()
    print(f"Stages 1-3 done: {len(res.tables)} tables, Figures 1-8 written.")


if __name__ == "__main__":
    main()

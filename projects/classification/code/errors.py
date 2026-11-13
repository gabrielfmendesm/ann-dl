"""Step 6 — error analysis of the chosen network (seed 42) on the test set, after the single evaluation of step 5:
errors per subgroup, the use of the clinical flags within age bands, the worst examples and permutation importance.
Nothing here feeds back into any choice. Tables 12-15, Figures 6-7.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from common import CLASS_COLORS, COST_MISS, FEATURES, SEED, Results, load_raw, save_figure
from network import predict_proba

N_REPEATS = 30            # permutations per feature
N_WORST = 5               # worst examples shown of each kind of error
GLUCOSE_VALLEY = 175      # mg/dL: the valley between the two glucose modes found by the EDA
ADULT, OLD = 18, 60       # the EDA's age bands: under 18, 18 to 59, 60 or more


def age_bands(age):
    """The EDA's three age bands, as masks."""
    return {f"under {ADULT}": age < ADULT, f"{ADULT}–{OLD - 1}": (age >= ADULT) & (age < OLD),
            f"{OLD} or more": age >= OLD}


def subgroups(X):
    """Subgroups taken from the EDA's findings — age bands, the clinical flags, BMI missingness, the glucose mode —
    and gender, which the EDA found unrelated to stroke, as a check that the errors do not fall on one group."""
    return {
        **{("age", band): mask for band, mask in age_bands(X["age"]).items()},
        ("hypertension", "no"): X["hypertension"] == 0, ("hypertension", "yes"): X["hypertension"] == 1,
        ("heart disease", "no"): X["heart_disease"] == 0, ("heart disease", "yes"): X["heart_disease"] == 1,
        ("BMI", "recorded"): X["bmi"].notna(), ("BMI", "missing"): X["bmi"].isna(),
        ("glucose", f"below {GLUCOSE_VALLEY}"): X["avg_glucose_level"] < GLUCOSE_VALLEY,
        ("glucose", f"{GLUCOSE_VALLEY} or more"): X["avg_glucose_level"] >= GLUCOSE_VALLEY,
        ("gender", "female"): X["gender"] == "Female", ("gender", "male"): X["gender"] == "Male",
    }


def plot_subgroups(table):
    """Figure 6: miss rate and false-alarm rate per subgroup."""
    labels = [f"{f}: {g}" for f, g in zip(table["feature"], table["subgroup"])]
    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9, 6.2))
    ax.barh(y - 0.2, table["miss rate (%)"], height=0.4, color=CLASS_COLORS[1], label="miss rate (% of its strokes)")
    ax.barh(y + 0.2, table["false-alarm rate (%)"], height=0.4, color=CLASS_COLORS[0],
            label="false-alarm rate (% of its non-strokes)")
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set(title="Test patients, the network at its validation threshold", xlabel="rate (%)", ylabel="subgroup")
    ax.legend(loc="lower right")
    save_figure(fig, 6, "Errors of the network by subgroup")


def plot_importance(table):
    """Figure 7: permutation importance (drop in test AP) of each recorded feature."""
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    y = np.arange(len(table))
    ax.barh(y, table["AP drop (mean)"], xerr=table["AP drop (sd)"], color=CLASS_COLORS[1], capsize=3,
            label=f"mean ± sd over {N_REPEATS} permutations")
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(y, table["feature"])
    ax.invert_yaxis()
    ax.set(title="Drop in test average precision when one recorded feature is shuffled", xlabel="AP drop",
           ylabel="feature (as recorded)")
    ax.legend(loc="lower right")
    save_figure(fig, 7, "Permutation importance of the network on the test set")


def clinical_flags(X):
    """The three recorded risk markers the EDA found associated with stroke beyond age."""
    return {"hypertension": X["hypertension"] == 1, "heart disease": X["heart_disease"] == 1,
            f"glucose {GLUCOSE_VALLEY} or more": X["avg_glucose_level"] >= GLUCOSE_VALLEY}


def run(X_test, y_test, final):
    """Step 6 (Tables 12-15, Figures 6-7), on the seed-42 network of step 5 at its validation threshold."""
    results = Results("errors")
    p, t = final["probabilities"]["Network"], final["threshold"]
    alarm = p >= t   # the network's decisions: True flags the patient

    rows = []
    for (feature, group), mask in subgroups(X_test).items():
        mask = mask.to_numpy()
        pos, neg = int((y_test[mask] == 1).sum()), int((y_test[mask] == 0).sum())
        fn, fp = int((~alarm & mask & (y_test == 1)).sum()), int((alarm & mask & (y_test == 0)).sum())
        rows.append({"feature": feature, "subgroup": group, "patients": int(mask.sum()), "strokes": pos, "missed": fn,
                     "false alarms": fp, "miss rate (%)": 100 * fn / pos if pos else np.nan,
                     "false-alarm rate (%)": 100 * fp / neg if neg else np.nan,
                     "error rate (%)": 100 * (fn + fp) / mask.sum(),
                     "cost per patient": (COST_MISS * fn + fp) / mask.sum()})
    table = pd.DataFrame(rows)
    results.table(12, "Errors of the network by subgroup on the test set, at the validation threshold", table,
                  floatfmt={"miss rate (%)": ".2f", "false-alarm rate (%)": ".2f", "error rate (%)": ".2f"})
    worst = table.loc[table["error rate (%)"].idxmax()]
    costliest = table.loc[table["cost per patient"].idxmax()]
    plot_subgroups(table)

    # Does the network use the clinical flags, or only age? The share of non-strokes it flags, with and without each
    # flag (and with any of them against none), inside each adult age band, with their median age: within a band the
    # patients who carry a flag are also older, so the comparison is not at equal age.
    age, negatives, rows = X_test["age"].to_numpy(), y_test == 0, []
    bands = age_bands(age)
    markers = {name: has.to_numpy() for name, has in clinical_flags(X_test).items()}
    any_marker = np.any(list(markers.values()), axis=0)
    for band in (f"{ADULT}–{OLD - 1}", f"{OLD} or more"):
        for name, has in {**markers, "any of the three": any_marker}.items():
            with_, without = bands[band] & negatives & has, bands[band] & negatives & ~has
            rows.append({"age": band, "flag": name, "non-strokes with it": int(with_.sum()),
                         "median age with it": np.median(age[with_]), "flagged with it (%)": 100 * alarm[with_].mean(),
                         "non-strokes without it": int(without.sum()), "median age without it": np.median(age[without]),
                         "flagged without it (%)": 100 * alarm[without].mean()})
    results.table(13, "Share of the non-strokes the network flags, with and without each clinical flag, by age band "
                      "(test set)", pd.DataFrame(rows),
                  floatfmt={"median age with it": ".1f", "flagged with it (%)": ".2f", "median age without it": ".1f",
                            "flagged without it (%)": ".2f"})
    strokes_18_59 = bands[f"{ADULT}–{OLD - 1}"] & (y_test == 1)
    old = bands[f"{OLD} or more"]
    hyper = X_test["hypertension"].to_numpy() == 1

    ids = load_raw()["id"]
    shown = ["age", "gender", "hypertension", "heart_disease", "avg_glucose_level", "bmi", "smoking_status"]
    missed = np.flatnonzero(~alarm & (y_test == 1))
    false_alarms = np.flatnonzero(alarm & (y_test == 0))
    picks = [("missed stroke", i) for i in missed[np.argsort(p[missed])][:N_WORST]] + \
            [("false alarm", i) for i in false_alarms[np.argsort(-p[false_alarms])][:N_WORST]]
    examples = pd.DataFrame([{"error": kind, "id": str(ids[X_test.index[i]]), **X_test.iloc[i][shown].to_dict(),
                              "probability": p[i]} for kind, i in picks])
    examples["bmi"] = examples["bmi"].map(lambda v: "missing" if pd.isna(v) else f"{v:.1f}")
    results.table(14, f"The worst errors: the {N_WORST} strokes with the lowest probability and the {N_WORST} "
                      "non-strokes with the highest", examples,
                  floatfmt={"age": "g", "avg_glucose_level": ".2f", "probability": ".4f"})

    model, preprocessor = final["model"], final["preprocessor"]
    base = average_precision_score(y_test, p)
    rng = np.random.default_rng(SEED)
    rows = []
    for feature in FEATURES:
        drops = []
        for _ in range(N_REPEATS):
            X = X_test.copy()
            X[feature] = rng.permutation(X[feature].to_numpy())
            drops.append(base - average_precision_score(y_test, predict_proba(model, preprocessor.transform(X))))
        rows.append({"feature": feature, "AP drop (mean)": np.mean(drops), "AP drop (sd)": np.std(drops, ddof=1)})
    importance = pd.DataFrame(rows).sort_values("AP drop (mean)", ascending=False, kind="stable").reset_index(drop=True)
    results.table(15, f"Permutation importance on the test set ({N_REPEATS} permutations per feature; "
                      f"test AP {base:.4f})", importance)
    plot_importance(importance)

    results.metrics.update({
        "worst_subgroup": f"{worst['feature']}: {worst['subgroup']}",
        "worst_error_rate": float(worst["error rate (%)"]),
        "worst_subgroup_row": worst.to_dict(), "costliest_subgroup_row": costliest.to_dict(),
        "subgroups": table.to_dict(orient="records"),
        "strokes_18_59": int(strokes_18_59.sum()),
        "strokes_18_59_with_clinical_flag": int((strokes_18_59 & any_marker).sum()),
        "strokes_18_59_with_clinical_flag_missed": int((strokes_18_59 & any_marker & ~alarm).sum()),
        "stroke_rate_60_plus": {"hypertension": float(y_test[old & hyper].mean()),
                                "no hypertension": float(y_test[old & ~hyper].mean()),
                                "n_hypertension": int((old & hyper).sum()),
                                "n_no_hypertension": int((old & ~hyper).sum())},
        "test_strokes_hypertension": int((hyper & (y_test == 1)).sum()),
        "importance": importance.to_dict(orient="records"), "base_ap": base,
    })
    results.write()
    print(table.to_string(index=False))
    print(examples.to_string(index=False))
    print(importance.to_string(index=False))

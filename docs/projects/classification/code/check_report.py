"""Check that the report quotes exactly what the scripts produced.

Usage, from the repository root, after main.py:
    python docs/projects/classification/code/check_report.py

Every table of results/*_tables.md must appear verbatim in index.md, as a whole table; every figure must be
committed, shown exactly once and have a "Conclusion — Figure N." paragraph; the key numbers of the text must be the
ones stored in results/ (the *_metrics.json files, split_sizes.json and five of the table CSVs), compared with the
report's typographic minus sign normalized to "-"; and the results summary must be the last section, with its eight
rows in order, each rebuilt here from those files and compared whole.
"""

import json
import re
import sys

import numpy as np
import pandas as pd

from common import CRITERION_GAIN, FIGURES_DIR, MAIN_METRIC, N_FOLDS, REPORT, RESULTS_DIR, SEED

STEPS = ("starting_point", "baselines", "search", "evaluation", "errors")
N_FIGURES = 7
SUMMARY_ROWS = ("Dataset, task, target and main metric", "Train · validation · test (rows)",
                "Best baseline and its test metric",
                "Chosen network: layers × units, number of parameters, regularization",
                "Optimizer, learning rate, batch and the epoch it stopped at",
                "The network's main metric on the test set (mean ± std, 3 seeds)", "Success criterion met?",
                "Subgroup with the largest error and its value")   # the statement's rows, in its order


def f4(x):
    return f"{x:.4f}"


def ci(interval):
    return f"[{interval[0]:.4f}, {interval[1]:.4f}]"


def listing(items):
    """'a, b and c', as the report enumerates values."""
    items = [str(i) for i in items]
    return f"{', '.join(items[:-1])} and {items[-1]}"


def power_of_ten(x):
    """1.1196e-09 -> '1.1 × 10⁻⁹', as the report writes small p-values."""
    mantissa, exponent = f"{x:.1e}".split("e")
    superscript = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")
    return f"{mantissa} × 10{str(int(exponent)).translate(superscript)}"


def main():
    report = REPORT.read_text()
    text = report.replace("−", "-")
    problems = []

    tables = 0
    for step in STEPS:
        generated = (RESULTS_DIR / f"{step}_tables.md").read_text()
        for number, body in re.findall(r"^### Table (\d+) — .*\n\n((?:\|.*\n)+)", generated, re.M):
            tables += 1
            if f"\n\n{body.strip()}\n\n" not in report:   # the whole table, with no row added or missing
                problems.append(f"Table {number} in the report differs from the one in results/{step}_tables.md")

    for n in range(1, N_FIGURES + 1):
        if not (FIGURES_DIR / f"fig{n}.png").is_file():
            problems.append(f"figures/fig{n}.png is missing")
        if report.count(f"](figures/fig{n}.png)") != 1:
            problems.append(f"Figure {n} is not shown exactly once")
        if f"**Conclusion — Figure {n}.**" not in report:
            problems.append(f"Figure {n} has no conclusion")

    m = {step: json.loads((RESULTS_DIR / f"{step}_metrics.json").read_text()) for step in STEPS}
    start, base, search, ev, err = (m[s] for s in STEPS)
    sizes = json.loads((RESULTS_DIR / "split_sizes.json").read_text())
    indicator = pd.read_csv(RESULTS_DIR / "table01.csv", index_col="model")
    validation_ap = pd.read_csv(RESULTS_DIR / "table02.csv", index_col="model")["AP (mean)"]
    regularization = pd.read_csv(RESULTS_DIR / "table05.csv", index_col="regularization")
    decay, gap = regularization["weights kept by the decay"], regularization["validation − training loss"]
    flags_by_age = pd.read_csv(RESULTS_DIR / "table13.csv", index_col=["age", "flag"])
    young_any = flags_by_age.loc[("18–59", "any of the three")]
    old_any = flags_by_age.loc[("60 or more", "any of the three")]
    test, seeds, boot, mcnemar = ev["test"], ev["seeds"], ev["bootstrap"], ev["mcnemar"]
    hgb, lr = test["Gradient boosting"]["ap"], test["Logistic regression"]["ap"]
    worst, costliest = err["worst_subgroup_row"], err["costliest_subgroup_row"]
    top, default = search["best_rank1"], search["default"]
    learned = ("Network", "Logistic regression", "Gradient boosting")
    flags = [100 * (test[k]["TP"] + test[k]["FP"]) / ev["test_rows"] for k in learned]
    later = [i for i in range(len(default["ap"])) if default["val_ap_max_epoch"][i] > default["best_epoch"][i]]
    earlier = [i for i in range(len(default["ap"])) if default["val_ap_max_epoch"][i] < default["best_epoch"][i]]
    given_up = [default["val_ap_max"][i] - default["ap"][i] for i in later + earlier]
    with_indicator = indicator.loc[["Logistic regression", "Gradient boosting"], "AP with the indicator"]
    other_seeds = [r["ap"] for seed, r in ev["per_seed"].items() if seed != str(SEED)]
    seed_cost = np.mean([r["cost_per_patient"] for r in ev["per_seed"].values()])
    lr_cost = test["Logistic regression"]["cost_per_patient"]
    cheaper_seeds = [(seed, r["cost_per_patient"]) for seed, r in ev["per_seed"].items()
                     if r["cost_per_patient"] < lr_cost]
    young, young_flag = err["strokes_18_59"], err["strokes_18_59_with_clinical_flag"]
    young_flag_caught = young_flag - err["strokes_18_59_with_clinical_flag_missed"]
    hypertension_60 = err["stroke_rate_60_plus"]
    gender = {r["subgroup"]: r["error rate (%)"] for r in err["subgroups"] if r["feature"] == "gender"}
    words = dict(enumerate(["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]))
    search_table = pd.read_csv(RESULTS_DIR / "table06.csv")
    eligible = search_table.loc[search_table["within one SE"] == "yes", "parameters"]
    smallest = eligible[eligible == eligible.min()]
    ap_of = search_table.set_index(["hidden layers", "learning rate", "regularization"])["AP (mean)"]
    decay_gain = {(h, rate): ap_of[(h, rate, "weight decay 1")] - ap_of[(h, rate, "none")]
                  for h in ("16", "32-16", "64-32") for rate in (0.001, 0.003)}
    larger_gains = [g for (h, _), g in decay_gain.items() if h != "16"]
    steps_final = -(-sizes["train"] // search["batch_size"])   # mini-batches per epoch on all training rows
    importance = {r["feature"]: r["AP drop (mean)"] for r in err["importance"]}
    # An entry whose condition fails expects a placeholder that the report never contains, so the check reports it.
    expected = {
        "split": f"{sizes['train']:,} training rows with {sizes['train_positives']} strokes, {sizes['test']:,} test "
                 f"rows with {sizes['test_positives']}",
        "fold sizes": f"{min(sizes['validation_folds'])} or {max(sizes['validation_folds'])} rows each, with "
                      f"{min(sizes['validation_positives'])} or {max(sizes['validation_positives'])} strokes",
        "indicator gains": f"logistic regression by {f4(start['gain']['Logistic regression'])} and of the network by "
                           f"{f4(start['gain']['Network (step 2)'])}",
        "indicator dropped": "the indicator is removed, for every model" if start["drop_indicator"]
                             else "<the indicator is kept>",
        "best with the indicator": f"with the indicator it would have been {with_indicator.idxmax().lower()} "
                                   f"({f4(with_indicator.max())})",
        "inputs": f"The pipeline now outputs {search['n_inputs']} columns; in fold 1 it outputs "
                  f"{search['fold_inputs'][0]}",
        "best baseline": f"**{base['best_baseline']} is the best baseline on validation**",
        "baseline lead": f"It leads logistic regression by "
                         f"{f4(validation_ap['Gradient boosting'] - validation_ap['Logistic regression'])}",
        "default params": f"**{search['default_params']:,} parameters for {sizes['train']:,} training rows**",
        "pos_weight range": f"{min(search['pos_weight_folds']):.2f} to {max(search['pos_weight_folds']):.2f} across "
                            f"the folds, {ev['pos_weight']:.2f} on the full training set",
        "rows per parameter": f"one parameter for every {words[round(sizes['train'] / search['chosen_params'])]} "
                              "training rows",
        "epochs run": f"stops between epochs {min(default['epochs_run'])} and {max(default['epochs_run'])}, far below "
                      f"the cap of {search['max_epochs']}",
        "training loss range": f"(to between {min(default['train_loss_last']):.4f} and "
                               f"{max(default['train_loss_last']):.4f})",
        "best epochs": f"first {min(default['best_epoch'])} to {max(default['best_epoch'])} epochs",
        "fold 5": f"fold 5 goes from {f4(default['val_loss_best'][4])} at its best epoch, "
                  f"{default['best_epoch'][4]}, to {f4(default['val_loss_last'][4])} at epoch "
                  f"{default['epochs_run'][4]}",
        "AP after the restored epoch": f"(to {listing(f4(default['val_ap_max'][i]) for i in later)} at epochs "
                                       f"{listing(default['val_ap_max_epoch'][i] for i in later)}, against "
                                       f"{listing(f4(default['ap'][i]) for i in later)} at the restored ones)",
        "AP before the restored epoch": f"in fold {earlier[0] + 1} it peaks earlier, at epoch "
                                        f"{default['val_ap_max_epoch'][earlier[0]]} "
                                        f"({f4(default['val_ap_max'][earlier[0]])} against "
                                        f"{f4(default['ap'][earlier[0]])})",
        "AP given up": f"gives up {min(given_up):.3f} to {max(given_up):.3f} of validation AP in "
                       f"{words[len(given_up)]} folds",
        "weights kept": f"the decay factor alone leaves the weights {f4(decay['weight decay 0.1'])} of their size",
        "decay factors": f"The factor falls to {f4(decay['weight decay 1'])}, the gap between validation and training "
                         f"loss at the restored epoch narrows from {f4(gap['none'])} to {f4(gap['weight decay 1'])}",
        "strong decay": f"The factor reaches {f4(decay['weight decay 10'])}, the two losses meet at a high level (gap "
                        f"{f4(gap['weight decay 10'])})",
        "configs": f"{search['n_configs']} configurations",
        "decay on larger networks": f"raises the mean AP of all four 32-16 and 64-32 networks (by "
                                    f"{min(larger_gains):.4f} to {max(larger_gains):.4f}"
                                    if min(larger_gains) > 0 else "<weight decay lowers a larger network>",
        "decay on 16 units": f"lowers it for both 16-unit networks (from {f4(ap_of[('16', 0.001, 'none')])} to "
                             f"{f4(ap_of[('16', 0.001, 'weight decay 1')])} at lr 10⁻³, and from "
                             f"{f4(ap_of[('16', 0.003, 'none')])} to {f4(ap_of[('16', 0.003, 'weight decay 1')])}"
                             if max(decay_gain[("16", r)] for r in (0.001, 0.003)) < 0 else "<16 units gain>",
        "one SE": f"{f4(top['AP (mean)'])} - {f4(top['AP (sd)'])}/√{N_FOLDS} ≈ {search['one_se_threshold']:.4f}",
        "smallest within one SE": f"Of these {search['n_within_one_se']}, {len(smallest)} have the smallest size, "
                                  f"{eligible.min()} parameters",
        "steps of the final retraining": f"an epoch has {steps_final} mini-batches instead of "
                                         f"{search['steps_per_epoch']}, so the {search['final_epochs']} epochs make "
                                         f"{steps_final * search['final_epochs']} steps against "
                                         f"{search['steps_per_epoch'] * search['final_epochs']} in a fold",
        "within one SE": f"{search['n_within_one_se']} of the {search['n_configs']} are within one standard error",
        "chosen": f"{search['chosen_params']} parameters** (rank {search['chosen_rank']}",
        "final epochs": f"restored (best) epochs in the five folds are "
                        f"{listing(search['chosen_cv']['best_epoch'])} — training ran {search['patience']} epochs "
                        f"past each — and the final network is retrained for their median, "
                        f"**{search['final_epochs']} epochs**",
        "low BMI": f"moves the validation AP by "
                   f"{np.mean(search['chosen_cv']['ap']) - np.mean(search['low_bmi_ap']):.4f}",
        "threshold": f"the cheapest cut is **{f4(ev['thresholds']['Network'])}**",
        "corrected threshold": f"w = {ev['pos_weight']:.2f}, the ratio of negatives to positives in training. "
                               f"Dividing the odds of {f4(ev['thresholds']['Network'])} by w gives odds of "
                               f"{f4(ev['threshold_corrected'] / (1 - ev['threshold_corrected']))}, a probability of "
                               f"{f4(ev['threshold_corrected'])}, close to p* = {f4(ev['p_star'])}",
        "threshold transfer": f"the network flags {flags[0]:.2f}% of the test patients against "
                              f"{100 * ev['validation_decisions']['Network']['flagged'] / sizes['train']:.2f}% of the "
                              "training rows on validation",
        "majority class": f"that costs {f4(test['Majority class']['cost_per_patient'])} per patient on the test "
                          f"set, slightly less than its own prediction, no stroke for anyone "
                          f"({f4(ev['trivial_costs']['flag nobody'])})"
                          if test["Majority class"]["FN"] == 0 else "<the majority class no longer flags everyone>",
        "flagged range": f"between {min(flags):.2f}% and {max(flags):.2f}% of the test patients",
        "McNemar vs LR": f"({mcnemar['Logistic regression']['only_first_right']} that only the network gets right, "
                         f"{mcnemar['Logistic regression']['only_second_right']} that only logistic regression "
                         f"does) gives p = {mcnemar['Logistic regression']['p']:.2f}",
        "McNemar vs HGB": f"the same test gives p = {power_of_ten(mcnemar['Gradient boosting']['p'])}",
        "criterion": f"**{f4(seeds['ap_mean'])} ± {f4(seeds['ap_sd'])}**",
        "criterion gain": f"The network is {100 * ev['criterion_gain']:.2f}% higher, more than the "
                          f"{100 * CRITERION_GAIN:.0f}% required, and the gap ({f4(seeds['ap_mean'] - hgb)})",
        "criterion met": "**the criterion, as written before training, is met.**" if ev["criterion_met"]
                         else "<the criterion is not met>",
        "logistic on test": f"scores {f4(lr)}: the network's mean is "
                            f"{-100 * ev['gain_vs']['Logistic regression']:.2f}% *below* it, a gap of "
                            f"{f4(seeds['ap_mean'] - lr)}, a fifth of the seeds' standard deviation"
                            if round(abs(seeds["ap_mean"] - lr) / seeds["ap_sd"], 1) == 0.2
                            else "<the gap to logistic regression is not a fifth of the sd>",
        "bootstrap network": f"the network's AP, mean of the 5 seeds: {ci(boot['network_ap_ci']['mean of the seeds'])}",
        "bootstrap vs LR": f"its gap to logistic regression: "
                           f"{ci(boot['diff']['mean of the seeds - Logistic regression']['ci'])}, positive in "
                           f"{100 * boot['diff']['mean of the seeds - Logistic regression']['share_positive']:.0f}% "
                           "of the resamples",
        "bootstrap vs HGB": f"its gap to gradient boosting: "
                            f"{ci(boot['diff']['mean of the seeds - Gradient boosting']['ci'])}",
        "bootstrap seed 42": f"for it alone, the gap to logistic regression is "
                             f"{ci(boot['diff']['seed 42 - Logistic regression']['ci'])}",
        "seed 42 against the others": f"(AP {f4(ev['per_seed'][str(SEED)]['ap'])} against {f4(min(other_seeds))} to "
                                      f"{f4(max(other_seeds))} for the others)",
        "worst subgroup": f"the largest error rate, {worst['error rate (%)']:.2f}%**: the network flags "
                          f"{worst['false alarms']} of their {worst['patients'] - worst['strokes']} non-strokes",
        "costliest subgroup": f"has the largest cost per patient, {f4(costliest['cost per patient'])} "
                              f"({costliest['missed']} missed stroke and {costliest['false alarms']} false alarms "
                              f"among {costliest['patients']} patients)",
        "strokes 18-59": f"Only {young_flag} of the {young} have a clinical flag, and {young_flag_caught} of those "
                         f"{young_flag} is caught",
        "strokes 18-59 without a flag": f"so the {young - young_flag} strokes without one",
        "flags at 18-59": f"{young_any['flagged with it (%)']:.2f}% of the non-strokes with one, against "
                          f"{young_any['flagged without it (%)']:.2f}% of those with none",
        "median ages at 18-59": f"have a median age of {young_any['median age with it']:.0f}, against "
                                f"{young_any['median age without it']:.0f} for those with none",
        "no flag at 60 or more": f"{old_any['flagged without it (%)']:.2f}% of their non-strokes with none of the "
                                 "three",
        "hypertension at 60 or more": f"{100 * hypertension_60['hypertension']:.2f}% against "
                                      f"{100 * hypertension_60['no hypertension']:.2f}% on the test set",
        "gender error rates": f"close error rates ({gender['female']:.2f}% and {gender['male']:.2f}%)",
        "hypertensive strokes": f"Only {err['test_strokes_hypertension']} of the strokes are hypertensive",
        "importance age": f"{err['importance'][0]['feature'].capitalize()} dominates, with an AP drop of "
                          f"{f4(err['importance'][0]['AP drop (mean)'])}",
        "importance low": f"BMI ({f4(importance['bmi'])}), gender ({f4(importance['gender'])}) and residence type "
                          f"({f4(importance['Residence_type'])}) rank low",
        "conclusion ROC-AUC": f"in ROC-AUC ({f4(test['Logistic regression']['auc'])} against "
                              f"{f4(seeds['auc_mean'])} ± {f4(seeds['auc_sd'])})",
        "conclusion cost": f"at a lower cost per patient ({f4(lr_cost)} against {f4(seed_cost)} for the network's "
                           f"seed mean; only seed {cheaper_seeds[0][0]}, at {f4(cheaper_seeds[0][1])}, is marginally "
                           "below it)" if len(cheaper_seeds) == 1 else "<a different number of seeds beat it>",
        "test against validation": f"(network {f4(seeds['ap_mean'])} against "
                                   f"{f4(np.mean(search['chosen_cv']['ap']))}, logistic regression {f4(lr)} against "
                                   f"{f4(validation_ap['Logistic regression'])}, gradient boosting {f4(hgb)} against "
                                   f"{f4(validation_ap['Gradient boosting'])})",
        "limitation interval": f"(95% bootstrap interval {ci(boot['network_ap_ci']['mean of the seeds'])} for the "
                               "seed mean)",
    }
    for name, value in expected.items():
        if value.replace("−", "-") not in text:
            problems.append(f"{name}: '{value}' not found in the report")

    # The results summary, rebuilt row by row from results/ and compared whole.
    chosen, other = search["chosen"], "Logistic regression"
    best = base["best_baseline"]
    layers = " → ".join([f"{search['n_inputs']} inputs", *[f"{u} ReLU units" for u in chosen["hidden"]], "1 logit"])
    no_regularization = chosen["weight_decay"] == 0 and chosen["dropout"] == 0
    values = [
        "Stroke Prediction Dataset (fedesoriano, Kaggle) · binary classification · target `stroke` "
        f"(1 = had a stroke) · {MAIN_METRIC}",
        f"train {sizes['train']:,} rows ({sizes['train_positives']} strokes), validated in {N_FOLDS} stratified folds "
        f"of {min(sizes['validation_folds'])} or {max(sizes['validation_folds'])} rows "
        f"({min(sizes['validation_positives'])} or {max(sizes['validation_positives'])} strokes) · test "
        f"{sizes['test']:,} rows ({sizes['test_positives']} strokes)",
        f"{best} (best validation AP, {f4(validation_ap[best])}): test AP {f4(test[best]['ap'])} · {other.lower()}, "
        f"second on validation ({f4(validation_ap[other])}): test AP {f4(test[other]['ap'])}"
        if best != other and validation_ap.drop(best).idxmax() == other else "<the baseline ranking changed>",
        f"{layers} ({words[len(chosen['hidden'])]} hidden layer of {chosen['hidden'][0]} units), "
        f"{search['chosen_params']} parameters, no weight decay and no dropout (early stopping)"
        if no_regularization and len(chosen["hidden"]) == 1 else "<the chosen network changed>",
        f"AdamW with weight decay 0 (that is, Adam), learning rate {chosen['lr']:g}, batch {search['batch_size']}; "
        f"early stopping restored epochs {listing(search['chosen_cv']['best_epoch'])} in the five folds, so the final "
        f"network trains for their median, {search['final_epochs']} epochs",
        f"average precision {f4(seeds['ap_mean'])} ± {f4(seeds['ap_sd'])} (mean ± sample sd over {len(seeds['ap'])} "
        f"seeds); ROC-AUC {f4(seeds['auc_mean'])} ± {f4(seeds['auc_sd'])}",
        f"Yes, as written: {100 * ev['criterion_gain']:.2f}% above gradient boosting, a gap of "
        f"{f4(seeds['ap_mean'] - hgb)} > sd {f4(seeds['ap_sd'])} · but the 95% bootstrap interval of that gap, "
        f"{ci(boot['diff']['mean of the seeds - Gradient boosting']['ci'])}, contains 0, and the network's mean is "
        f"{-100 * ev['gain_vs']['Logistic regression']:.2f}% below logistic regression (test AP {f4(lr)}): the network "
        "is not better than the linear model" if ev["criterion_met"] else "<the criterion is not met>",
        f"{worst['feature'].capitalize()} {worst['subgroup']}: error rate {worst['error rate (%)']:.2f}% "
        f"({worst['false alarms']} false alarms among its {worst['patients'] - worst['strokes']} non-strokes, "
        f"{worst['missed']} of its {worst['strokes']} strokes missed) · by cost per patient, {costliest['subgroup']} "
        f"{costliest['feature']}: {f4(costliest['cost per patient'])} ({costliest['missed']} of "
        f"{costliest['strokes']} strokes missed, {costliest['false alarms']} false alarms among "
        f"{costliest['patients']} patients)",
    ]
    rebuilt = ["| # | Summary | Value |", "|---|---|---|",
               *(f"| {i} | {row} | {value} |" for i, (row, value) in enumerate(zip(SUMMARY_ROWS, values), start=1))]
    section = report.split("\n## Results summary\n", 1)[-1].strip().splitlines()
    if "\n## Results summary\n" not in report or "\n## " in "\n".join(section):
        problems.append("the results summary is missing or is not the last section")
    for i, (line, want) in enumerate(zip(section, rebuilt)):
        if line.replace("−", "-") != want.replace("−", "-"):
            problems.append(f"results summary, line {i + 1}: expected '{want}'")
    if len(section) != len(rebuilt):
        problems.append(f"the results summary has {len(section) - 2} rows instead of {len(SUMMARY_ROWS)}")

    if problems:
        print("\n".join(problems))
        sys.exit(1)
    print(f"Report consistent with results/: {tables} tables, {N_FIGURES} figures, {len(expected)} quoted values and "
          f"the {len(SUMMARY_ROWS)} rows of the results summary.")


if __name__ == "__main__":
    main()

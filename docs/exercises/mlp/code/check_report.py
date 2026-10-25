"""Check that the numbers in the report are the numbers the code produces.

Run after main.py: it reads results/results.json and confirms that every key
value appears in index.md exactly as computed (with the report's typographic
minus sign normalized to "-"). Any mismatch is listed and the script fails.
"""

import json
import sys

from common import HERE

REPORT = (HERE.parent / "index.md").read_text().replace("−", "-")
results = json.loads((HERE.parent / "results" / "results.json").read_text())
ex1, ex2, ex3, ex4 = results["ex1"], results["ex2"], results["ex3"], results["ex4"]


def pct(x):
    return f"{100 * x:.2f}%"


expected = {
    # Exercise 1: every value of the forward pass and every updated parameter, 6 decimals
    **{f"z1[{i}]": f"{v:.6f}" for i, v in enumerate(ex1["z1"])},
    **{f"h1[{i}]": f"{v:.6f}" for i, v in enumerate(ex1["h1"])},
    "u2": f"{ex1['u2']:.6f}", "y_hat": f"{ex1['y_hat']:.6f}", "loss": f"{ex1['loss']:.6f}",
    **{f"W1[{i}][{j}]": f"{v:.6f}" for i, row in enumerate(ex1["W1"]) for j, v in enumerate(row)},
    **{f"b1[{i}]": f"{v:.6f}" for i, v in enumerate(ex1["b1"])},
    **{f"W2[{i}]": f"{v:.6f}" for i, v in enumerate(ex1["W2"])},
    "b2": f"{ex1['b2']:.6f}", "loss after the update": f"L = {ex1['loss_after']:.4f}",
    # Exercise 2
    "Ex. 2 baseline": pct(ex2["baseline_accuracy"]), "Ex. 2 test accuracy": f"test {pct(ex2['accuracy'])}",
    "Ex. 2 parameters": f"{ex2['n_parameters']} parameters", "Ex. 2 final loss": f"{ex2['final_loss']:.4f} at epoch 2000",
    # Exercise 3
    "Ex. 3 test accuracy": f"test {pct(ex3['accuracy'])}", "Ex. 3 baseline": f"**{pct(ex3['baseline_accuracy'])}**",
    "Ex. 3 parameters": f"{ex3['n_parameters']} parameters", "Ex. 3 final loss": f"{ex3['final_loss']:.4f} at epoch 2000",
    "Ex. 3 most-confused pair": f"Class {ex3['worst_pair'][0]} and Class {ex3['worst_pair'][1]}",
    "Ex. 3 gain over the baseline": f"{100 * (ex3['accuracy'] - ex3['baseline_accuracy']):.2f} points",
    # Exercise 4
    "Ex. 4 seed-42 accuracy": f"**{pct(ex4['deep_accuracy'])}**",
    "Ex. 4 shallow mean": f"{pct(ex4['shallow']['test_mean'])} ± {pct(ex4['shallow']['test_std'])}",
    "Ex. 4 deep mean": f"{pct(ex4['deep']['test_mean'])} ± {pct(ex4['deep']['test_std'])}",
    "Ex. 4 shallow loss": f"{ex4['shallow']['loss_mean']:.4f} ± {ex4['shallow']['loss_std']:.4f}",
    "Ex. 4 deep loss": f"{ex4['deep']['loss_mean']:.4f} ± {ex4['deep']['loss_std']:.4f}",
    "Ex. 4 parameters": f"| {ex4['shallow']['n_parameters']} | ",
    "Ex. 4 deep parameters": f"| {ex4['deep']['n_parameters']} | ",
    "Ex. 4 mean difference": f"{100 * ex4['diff_mean']:.2f} points",
    "Ex. 4 seeds": f"{len(ex4['seeds'])} seeds",
}
# The Results summary rows, value by value (the report uses superscripts for z(1), h(1), u(2), W(1), ...)
superscripts = {"(1)": "⁽¹⁾", "(2)": "⁽²⁾", "y_hat": "ŷ"}
for row in results["summary_rows"]:
    value = row["value"]
    for plain, pretty in superscripts.items():
        value = value.replace(plain, pretty)
    expected[f"summary row {row['#']}"] = value.replace("−", "-")

missing = {name: text for name, text in expected.items() if text not in REPORT}
print(f"{len(expected) - len(missing)} of {len(expected)} values found in index.md")
for name, text in missing.items():
    print(f"  MISSING {name}: {text!r}")
sys.exit(1 if missing else 0)

"""Reproduces every number and figure of the MLP report.

Run from the repository root:

    python docs/exercises/mlp/code/main.py

Every network starts from np.random.default_rng(seed) with seed 42 (Exercise 4
repeats the comparison for seeds 42-51); the datasets and the train/test splits
use random_state=42. The results are also written to results/results.json, which
check_report.py compares with the numbers in the report.
"""

import json

import numpy as np

import exercise1
import exercise2
import exercise3
import exercise4
import gradient_check
from common import HERE


def pct(x):
    return f"{100 * x:.2f}%"


ex1 = exercise1.run()
max_gradient_error = gradient_check.run()
ex2 = exercise2.run()
ex3 = exercise3.run()
ex4 = exercise4.run(ex3)

shallow, deep = ex4["summary"]["shallow"], ex4["summary"]["deep"]
i, j = ex3["worst_pair"]
rows = [
    ("Ex. 1 — z(1) and h(1)", f"z(1) = [{ex1['z1'][0]:.4f}, {ex1['z1'][1]:.4f}], h(1) = [{ex1['h1'][0]:.4f}, {ex1['h1'][1]:.4f}]"),
    ("Ex. 1 — u(2), y_hat and L", f"u(2) = {ex1['u2']:.4f}, y_hat = {ex1['y_hat']:.4f}, L = {ex1['loss']:.4f}"),
    ("Ex. 1 — the eight updated parameters",
     f"W(1) = [[{ex1['W1'][0, 0]:.4f}, {ex1['W1'][0, 1]:.4f}], [{ex1['W1'][1, 0]:.4f}, {ex1['W1'][1, 1]:.4f}]], "
     f"b(1) = [{ex1['b1'][0]:.4f}, {ex1['b1'][1]:.4f}], W(2) = [{ex1['W2'][0]:.4f}, {ex1['W2'][1]:.4f}], b(2) = {ex1['b2']:.4f}"),
    ("Ex. 2 — linear baseline test accuracy", pct(ex2["baseline_accuracy"])),
    ("Ex. 2 — architecture and test accuracy", f"{ex2['layers']}, {pct(ex2['accuracy'])}"),
    ("Ex. 3 — architecture and test accuracy", f"{ex3['layers']}, {pct(ex3['accuracy'])}"),
    ("Ex. 3 — most-confused class pair", f"Class {i} and Class {j}"),
    ("Ex. 4 — architecture and test accuracy", f"{ex4['deep_layers']}, {pct(ex4['deep_accuracy'])}"),
    ("Ex. 4 — accuracy difference against Ex. 3 (mean over seeds)",
     f"{100 * ex4['diff_mean']:+.2f} points ({pct(deep['test_mean'])} vs {pct(shallow['test_mean'])}, {len(ex4['seeds'])} seeds)"),
]
print("\n" + "=" * 72 + "\nRESULTS SUMMARY\n" + "=" * 72)
for n, (quantity, value) in enumerate(rows, start=1):
    print(f"{n} | {quantity} | {value}")

results = {
    "ex1": {k: np.asarray(v).round(6).tolist() for k, v in ex1.items()},
    "max_gradient_check_error": max_gradient_error,
    "ex2": ex2,
    "ex3": {k: v for k, v in ex3.items() if k not in ("history", "confusion")} | {"confusion": ex3["confusion"].tolist()},
    "ex4": {"deep_layers": ex4["deep_layers"], "deep_accuracy": ex4["deep_accuracy"], "seeds": ex4["seeds"],
            "diff_mean": ex4["diff_mean"], "diff_std": ex4["diff_std"],
            **{name: {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in s.items() if k != "curves"}
               for name, s in ex4["summary"].items()}},
    "summary_rows": [{"#": n, "quantity": q, "value": v} for n, (q, v) in enumerate(rows, start=1)],
}
results_dir = HERE.parent / "results"
results_dir.mkdir(exist_ok=True)
(results_dir / "results.json").write_text(json.dumps(results, indent=2, default=float) + "\n")

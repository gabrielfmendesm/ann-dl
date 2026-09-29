"""Reproduces every number and figure of the Perceptron report.

Run from the repository root:

    python docs/exercises/perceptron/code/main.py

A single generator, created here with the fixed seed, is shared by the whole
report and consumed in this order: Exercise 1 data (Class 0, Class 1, the
presentation order) and initial weights, then the same for Exercise 2.
"""

import numpy as np

import exercise1
import exercise2

rng = np.random.default_rng(42)

ex1 = exercise1.run(rng)
ex2 = exercise2.run(rng)


def fmt(w, b):
    return f"w = [{w[0]:.4f}, {w[1]:.4f}], b = {b:.4f}"


print("\n" + "=" * 72 + "\nRESULTS SUMMARY\n" + "=" * 72)
rows = [
    ("Exercise 1 — final w and b", fmt(ex1["w"], ex1["b"])),
    ("Exercise 1 — epochs to convergence", f"{ex1['epochs']}"),
    ("Exercise 1 — final accuracy", f"{ex1['accuracy']:.2%}"),
    ("Exercise 1 — epochs and final accuracy with eta = 1.0", f"{ex1['epochs_large']} epochs, {ex1['accuracy_large']:.2%}"),
    ("Exercise 2 — final w and b", fmt(ex2["final_w"], ex2["final_b"])),
    ("Exercise 2 — accuracy of the final weights", f"{ex2['final_accuracy']:.2%}"),
    ("Exercise 2 — accuracy of the pocket weights", f"{ex2['pocket_accuracy']:.2%}"),
    ("Exercise 2 — epoch at which the pocket best occurred", f"{ex2['pocket_epoch']}"),
]
for i, (quantity, value) in enumerate(rows, start=1):
    print(f"{i} | {quantity} | {value}")

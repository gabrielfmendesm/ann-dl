"""Reproduces every number and figure of the VAE report.

Run from the repository root:

    python docs/exercises/vae/code/main.py

MNIST is downloaded into data/MNIST on the first run. Each model is built right after torch.manual_seed(42),
the split uses torch.Generator().manual_seed(42), and the prior samples of Figure 3 are drawn after
torch.manual_seed(0). The results are also written to results/results.json, which check_report.py compares
with the numbers in the report.
"""

import json

import torch
import torchvision

import exercise1
import exercise2
import loss_check
from common import HERE, load_mnist

data = load_mnist()
ex1, model16 = exercise1.run(data)
checks = loss_check.run(model16, data["val"][0])
ex2 = exercise2.run(data, model16, ex1)

a, b = ex2["interp_indices"]
da, db = ex2["interp_digits"]
rows = [
    ("Ex. 1 — test reconstruction, KL and negative ELBO (d = 16)",
     f"reconstruction {ex1['test_rec']:.2f}, KL {ex1['test_kl']:.2f}, negative ELBO {ex1['test_neg_elbo']:.2f} "
     f"nats per image"),
    ("Ex. 1 — validation KL at the first and at the last epoch",
     f"{ex1['val_kl_first']:.2f} nats after epoch 1, {ex1['val_kl_last']:.2f} nats after epoch 20"),
    ("Ex. 2 — test reconstruction and KL (d = 2)",
     f"reconstruction {ex2['test_rec']:.2f}, KL {ex2['test_kl']:.2f} nats per image"),
    ("Ex. 2 — the two test indices interpolated", f"{a} (a {da}) and {b} (a {db})"),
]
print("\n" + "=" * 72 + "\nRESULTS SUMMARY\n" + "=" * 72)
for n, (quantity, value) in enumerate(rows, start=1):
    print(f"{n} | {quantity} | {value}")

results = {
    "versions": {"torch": torch.__version__, "torchvision": torchvision.__version__},
    "ex1": ex1,
    "loss_check": checks,
    "ex2": ex2,
    "summary_rows": [{"#": n, "quantity": q, "value": v} for n, (q, v) in enumerate(rows, start=1)],
}
results_dir = HERE.parent / "results"
results_dir.mkdir(exist_ok=True)
(results_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n")

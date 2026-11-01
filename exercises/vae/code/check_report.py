"""Check that the numbers in the report are the numbers the code produces.

Run after main.py: it reads results/results.json and confirms that every key value appears in index.md
exactly as computed (with the report's typographic minus sign normalized to "-"). Any mismatch is listed
and the script fails.
"""

import json
import math
import sys

from common import HERE

REPORT = (HERE.parent / "index.md").read_text().replace("−", "-")
results = json.loads((HERE.parent / "results" / "results.json").read_text())
ex1, ex2, checks = results["ex1"], results["ex2"], results["loss_check"]
h1 = ex1["history"]


def f2(x):
    return f"{x:.2f}"


def sci(x):
    """1.234e-08 -> '1.2 × 10⁻⁸', the report's notation."""
    mantissa, exponent = f"{x:.1e}".split("e")
    digits = str(abs(int(exponent))).translate(str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹"))
    return f"{mantissa} × 10{'⁻' if int(exponent) < 0 else ''}{digits}"


def epoch_row(e):
    """Row of the training table for epoch e (1-based)."""
    tr, tk = h1["train_rec"][e - 1], h1["train_kl"][e - 1]
    vr, vk = h1["val_rec"][e], h1["val_kl"][e]
    return f"| {e} | {f2(tr)} | {f2(tk)} | {f2(vr)} | {f2(vk)} | {f2(vr + vk)} |"


def nearest(values, v):
    """1-based position of the grid value closest to v."""
    return min(range(len(values)), key=lambda k: abs(values[k] - v)) + 1


z = ex2["grid_z"]
labels = ex2["grid_labels"]
i, j = ex2["overlap_pair"]
centre = ex2["overlap_centre"]
switch_rows = [r for r in range(len(labels)) if labels[r][10] == i and labels[r][11] == j]  # columns 11 -> 12
contiguous = switch_rows == list(range(switch_rows[0], switch_rows[-1] + 1))
boundary = ex2["boundary_cells"]
ink, path_grey, pixel_grey = ex2["ink_area"], ex2["grey_latent_row"], ex2["grey_pixel_row"]
same_ends = f"{100 * pixel_grey[0]:.1f}" == f"{100 * pixel_grey[-1]:.1f}"  # the report gives one value for both
spread = ex2["within_digit_variance"]
kl_dims = ex1["kl_per_dim"]
pairs = ex2["top_pairs"]
a, b = ex2["interp_indices"]
da, db = ex2["interp_digits"]

expected = {
    # Approach and Challenges
    "versions": f"torch {results['versions']['torch']} and torchvision {results['versions']['torchvision']}",
    "BCE value check": f"relative {sci(checks['bce_value_rel_error'])} on all 10 000 validation images",
    "BCE gradient check": f"gradient agrees to {sci(checks['bce_grad_rel_error_regular'])}",
    "floored pixels": f"on {100 * checks['n_floored'] / checks['n_pixels']:.2f}% of the pixels, all but "
                      f"{checks['n_floored_ink']} of them background pixels",
    "KL Monte Carlo": f"within {checks['kl_mc_worst_stderr']:.2f} standard errors "
                      f"({100 * checks['kl_mc_worst_rel']:.2f}%)",
    "reparameterization exact": "exactly" if checks["reparam_grad_error"] == 0 else "MISMATCH",
    "sum over mean": f"divide the reconstruction term by {checks['sum_over_mean_ratio']:.2f}",
    # Exercise 1
    "parameters": f"{ex1['n_parameters']:,} parameters".replace(",", " "),
    "untrained": f"validation reconstruction of {f2(ex1['val_rec_untrained'])} nats and a KL of "
                 f"{ex1['val_kl_untrained']:.4f} nats",
    **{f"epoch {e} row": epoch_row(e) for e in (1, 2, 5, 10, 15, 20)},
    "test reconstruction": f"| Reconstruction | **{f2(ex1['test_rec'])}** |",
    "test KL": f"| KL | **{f2(ex1['test_kl'])}** |",
    "test negative ELBO": f"| Negative ELBO (their sum) | **{f2(ex1['test_neg_elbo'])}** |",
    "BCE floor": f"{f2(ex1['bce_floor_test'])} nats per test image",
    "above the floor": f"{f2(ex1['test_rec'])} nats are {f2(ex1['test_rec'] - ex1['bce_floor_test'])} above",
    "epoch 1 train vs val": f"({f2(h1['train_rec'][0])} against {f2(h1['val_rec'][1])})",
    "final gap": f"{f2(h1['val_rec'][-1] - h1['train_rec'][-1])} nats above the training average",
    "Figure 2 indices": "test indices " + ", ".join(str(k) for k in ex1["first_test_index_per_digit"][:-1])
                        + f" and {ex1['first_test_index_per_digit'][-1]}",
    "untrained KL in 1E": f"(KL {ex1['val_kl_untrained']:.4f} nats)",
    "code-free decoder": f"still costs {f2(ex1['rec_no_code'])} nats",
    "KL rise": f"rose by {f2(ex1['val_kl_last'] - ex1['val_kl_first'])} nats ({f2(ex1['val_kl_first'])} to "
               f"{f2(ex1['val_kl_last'])})",
    "reconstruction fall": f"fell by {f2(ex1['val_rec_first'] - ex1['val_rec_last'])} ({f2(ex1['val_rec_first'])} "
                           f"to {f2(ex1['val_rec_last'])})",
    "total fall": f"total fell by "
                  f"{f2(ex1['val_rec_first'] + ex1['val_kl_first'] - ex1['val_rec_last'] - ex1['val_kl_last'])}",
    "active dimensions": f"{ex1['n_active']} of the 16 dimensions carry between {ex1['min_kl_active']:.3f} and "
                         f"{max(kl_dims):.3f} nats",
    "active sigma": f"mean σ between {ex1['sigma_active_range'][0]:.2f} and {ex1['sigma_active_range'][1]:.2f}",
    "inactive dimensions": f"the other {16 - ex1['n_active']} stay on the prior, with at most "
                           f"{ex1['max_kl_inactive']:.4f} nats and a mean σ between "
                           f"{ex1['sigma_inactive_range'][0]:.2f} and {ex1['sigma_inactive_range'][1]:.2f}",
    "log N": f"log 50 000 = {f2(ex1['log_n_train'])}",
    "mean training KL and bound": f"mean training KL of {f2(ex1['train_kl_end'])} nats at least "
                                  f"{f2(ex1['aggregate_kl_lower_bound'])} nats",
    "log q of the test codes": f"median log q(z) of {f2(ex1['logq_test_median'])} and a 1st percentile of "
                               f"{f2(ex1['logq_test_p01'])}",
    "log q of Figure 3": f"the median is {f2(ex1['logq_prior_median'])}, "
                         f"{f2(ex1['logq_test_median'] - ex1['logq_prior_median'])} nats lower, and "
                         f"{ex1['prior_codes_below_test_p01']} of the 64",
    "expected count": f"1% of 64 would be {0.01 * 64:.2f}",
    "grey pixels": f"{100 * ex1['grey_test']:.2f}% in the test images, {100 * ex1['grey_recon_test']:.2f}% in their "
                   f"reconstructions from μ and {100 * ex1['grey_prior_samples']:.2f}% in the samples of Figure 3",
    "why mu": f"{f2(ex1['test_rec_from_mu'])} nats on the test set against {f2(ex1['test_rec'])} with a sampled z",
    # Exercise 2
    "d = 2 validation": f"validation reconstruction is {f2(ex2['history']['val_rec'][-1])} nats and its KL "
                        f"{f2(ex2['history']['val_kl'][-1])} nats",
    "table d = 16": f"| d = 16 (Exercise 1) | {f2(ex1['test_rec'])} | {f2(ex1['test_kl'])} | "
                    f"{f2(ex1['test_neg_elbo'])} |",
    "table d = 2": f"| d = 2 | **{f2(ex2['test_rec'])}** | **{f2(ex2['test_kl'])}** | {f2(ex2['test_neg_elbo'])} |",
    "d = 2 differences": f"{f2(ex2['test_rec'] - ex1['test_rec'])} nats worse and the KL "
                         f"{f2(ex1['test_kl'] - ex2['test_kl'])} nats smaller",
    "grid edge": f"|z| ≤ {ex2['grid_edge']:.2f}",
    "grid range": f"from {f2(z[0])} to {f2(z[-1])}",
    "cell mass": f"{math.sqrt(ex2['cell_mass']):.4f}² = {ex2['cell_mass']:.4f}",  # the step in q, squared
    "interpolation indices": f"test index {a} (a {da}) and test index {b} (a {db})",
    "overlap pairs": "; ".join(f"{p['pair'][0]} and {p['pair'][1]}, {p['count']}" for p in pairs[:5]),
    "largest overlap": f"between **{i} and {j}**",
    "overlap centre": f"μ = ({f2(centre[0])}, {f2(centre[1])})",
    "boundary band": f"from row {boundary[0][0]}, column {boundary[0][1]} down to row {boundary[-1][0]}, "
                     f"column {boundary[-1][1]}",
    "switch rows": (f"rows {switch_rows[0] + 1} to {switch_rows[-1] + 1} the switch happens between columns 11 and "
                    f"12, from z₁ = {f2(z[10])} to {f2(z[11])}") if contiguous else "rows not contiguous",
    "mixed codes cell": f"around row {len(z) + 1 - nearest(z, centre[1])}, column {nearest(z, centre[0])}",
    "bits": f"{f2(ex2['test_kl'])} nats, that is {ex2['kl_bits_d2']:.1f} bits, against {ex2['kl_bits_d16']:.1f} bits",
    "digit bits": f"log₂ 10 = {math.log2(10):.1f} bits",
    "spread": f"only {spread['d2'] / spread['images']:.0%} as much as the images themselves (summed pixel variance "
              f"{f2(spread['d2'])} against {f2(spread['images'])}), against {spread['d16'] / spread['images']:.0%} "
              f"({f2(spread['d16'])})",
    "k-NN accuracy": f"{100 * ex2['knn_acc_d2']:.2f}% of the test images in 2-D and "
                     f"{100 * ex2['knn_acc_d16']:.2f}% in 16-D",
    "cells per digit": f"between {min(ex2['grid_label_counts'])} and {max(ex2['grid_label_counts'])} of the 225 cells",
    "pixel row ink": f"{min(ink['pixel_row'][2:6])} to {max(ink['pixel_row'][2:6])} ink pixels",
    "union and endpoints": f"close to the {ink['union']} that the {da} and the {db} cover together, while the {da} "
                           f"has {ink['x_a']}, the {db} has {ink['x_b']} and a test image has {ink['test_mean']:.1f}",
    "pixel row grey": (f"{100 * min(pixel_grey[2:5]):.1f}% to {100 * max(pixel_grey[2:5]):.1f}% at t = 0.22 to "
                       f"0.44, against {100 * pixel_grey[0]:.1f}% at both ends") if same_ends else "ends differ",
    "latent path density": f"lowest log q(z), {f2(ex2['logq_latent_path_min'])}, is far above the 1st percentile of "
                           f"the test codes, {f2(ex1['logq_test_p01'])}",
    "latent row 3": "at t = 0.56 the majority digit of the 10 nearest training images is a 3"
                    if ex2["latent_row_digits"][5] == 3 and f"{ex2['t'][5]:.2f}" == "0.56" else "NOT A 3",
    "latent row ink": f"between {min(ink['latent_row'])} and {max(ink['latent_row'])} pixels",
    "latent row grey": f"{100 * min(path_grey):.1f}% to {100 * max(path_grey):.1f}% grey pixels",
}
for row in results["summary_rows"]:
    expected[f"summary row {row['#']}"] = f"| {row['#']} | {row['quantity']} | {row['value']} |"

missing = {name: text for name, text in expected.items() if text.replace("−", "-") not in REPORT}
print(f"{len(expected) - len(missing)} of {len(expected)} values found in index.md")
for name, text in missing.items():
    print(f"  MISSING {name}: {text!r}")
sys.exit(1 if missing else 0)

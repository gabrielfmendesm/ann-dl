"""Checks of the hand-written parts of vae.py against independent computations.

1. The reconstruction term equals F.binary_cross_entropy(..., reduction="sum") divided by the batch size, the
   call the statement names, in value; its gradient, autograd's exact derivative, equals the built-in's
   wherever the built-in does not floor x_hat (1 - x_hat) at 1e-12; at outputs saturated at exactly 0 or 1 both
   give the same value and a zero gradient at the logits (and reduction="mean" is 784 times smaller).
2. The closed-form KL equals a Monte Carlo estimate of E_q[log q(z) - log p(z)], with both log-densities
   written out, within the estimate's standard error.
3. The reparameterization passes the gradients dz/dmu = 1 and dz/dlogvar = sigma * eps / 2.
"""

import math

import torch
import torch.nn.functional as F

from common import VAL_SEED
from vae import N_PIXELS, reconstruction_term, reparameterize

MC_SAMPLES = 200_000
BUILTIN_FLOOR = 1e-12  # the floor F.binary_cross_entropy's backward puts on x_hat (1 - x_hat)


def log_normal(z, mu, logvar):
    """log N(z; mu, diag(exp(logvar))), summed over the last dimension."""
    return -0.5 * (math.log(2 * math.pi) + logvar + (z - mu) ** 2 / logvar.exp()).sum(dim=-1)


def run(model, X_val):
    print("\n" + "=" * 72 + "\nLOSS CHECK — the hand-written terms against independent computations\n" + "=" * 72)
    model.eval()
    generator = torch.Generator().manual_seed(VAL_SEED)

    # --- 1. reconstruction term vs F.binary_cross_entropy ----------------------------------------
    # Values: the same formula, the same -100 floor on each log. Gradients: autograd differentiates our formula
    # exactly, (x_hat - x) / (x_hat (1 - x_hat)) per pixel; the built-in's backward divides by
    # max(x_hat (1 - x_hat), 1e-12) instead, so the two can only differ where that product is below 1e-12.
    value_error = grad_error = builtin_error = 0.0
    n_pixels = n_floored = n_floored_ink = 0
    for start in range(0, len(X_val), 1000):
        x = X_val[start:start + 1000]
        with torch.no_grad():
            x_hat, _, _ = model(x, generator)
        ours_in = x_hat.clone().requires_grad_(True)
        ref_in = x_hat.clone().requires_grad_(True)
        ours = reconstruction_term(ours_in, x)
        ref = F.binary_cross_entropy(ref_in, x, reduction="sum") / len(x)
        ours.backward()
        ref.backward()
        value_error = max(value_error, abs(ours.item() - ref.item()) / ref.item())
        product = x_hat * (1 - x_hat)
        regular = product >= BUILTIN_FLOOR
        grad_error = max(grad_error, ((ours_in.grad - ref_in.grad)[regular].abs().max()
                                      / ref_in.grad[regular].abs().max()).item())
        builtin = (x_hat - x) / product.clamp(min=BUILTIN_FLOOR) / len(x)
        builtin_error = max(builtin_error, ((ref_in.grad - builtin).abs().max() / builtin.abs().max()).item())
        n_pixels += x.numel()
        n_floored += int((~regular).sum())
        n_floored_ink += int((~regular & (x > 0)).sum())
    # Saturation: logits of +-200 make the float32 sigmoid return exactly 1 and 0 (x is the last chunk of the
    # loop). Both versions floor each log at -100; the gradient that reaches the logits must be finite and equal.
    logits_ours = torch.where(torch.arange(N_PIXELS) % 2 == 0, 200.0, -200.0).expand_as(x).clone().requires_grad_()
    logits_ref = logits_ours.detach().clone().requires_grad_()
    edge_ours = reconstruction_term(torch.sigmoid(logits_ours), x)
    edge_ref = F.binary_cross_entropy(torch.sigmoid(logits_ref), x, reduction="sum") / len(x)
    edge_ours.backward()
    edge_ref.backward()
    edge_grad_finite = bool(torch.isfinite(logits_ours.grad).all())
    edge_grad_error = (logits_ours.grad - logits_ref.grad).abs().max().item()
    edge_grad_max = logits_ours.grad.abs().max().item()
    edge_ours, edge_ref = edge_ours.item(), edge_ref.item()
    ratio = (F.binary_cross_entropy(x_hat, x, reduction="sum") / len(x)
             / F.binary_cross_entropy(x_hat, x, reduction="mean")).item()
    print(f"reconstruction term vs F.binary_cross_entropy(sum) / batch, on the {len(X_val)} validation images "
          f"(trained d = 16 model):")
    print(f"  value: largest relative difference {value_error:.1e}")
    print(f"  gradient: largest relative difference {grad_error:.1e} on the pixels with x_hat (1 - x_hat) >= "
          f"{BUILTIN_FLOOR:g}; the built-in's gradient is (x_hat - x) / max(x_hat (1 - x_hat), {BUILTIN_FLOOR:g}) "
          f"(relative difference {builtin_error:.1e}),")
    print(f"  so it is smaller than the exact derivative on the {n_floored} of {n_pixels} pixels "
          f"({n_floored / n_pixels:.2%}) below that floor, {n_floored_ink} of them with ink (x > 0)")
    print(f"  outputs saturated at exactly 0 and 1: {edge_ours:.4f} vs {edge_ref:.4f}; gradient at the logits "
          f"finite: {edge_grad_finite}, largest {edge_grad_max:.1e}, largest difference {edge_grad_error:.1e}")
    print(f"  sum / batch over reduction='mean': {ratio:.2f} (= {N_PIXELS} pixels)")

    # --- 2. closed-form KL vs Monte Carlo --------------------------------------------------------
    with torch.no_grad():
        mu, logvar = model.encode(X_val[:5])
    mc_gen = torch.Generator().manual_seed(0)
    worst_z = worst_rel = 0.0
    for m, lv in zip(mu, logvar):
        closed = (0.5 * (m ** 2 + lv.exp() - 1 - lv)).sum().item()
        z = m + torch.exp(0.5 * lv) * torch.randn(MC_SAMPLES, len(m), generator=mc_gen)
        terms = log_normal(z, m, lv) - log_normal(z, torch.zeros_like(m), torch.zeros_like(lv))
        estimate, stderr = terms.mean().item(), terms.std().item() / math.sqrt(MC_SAMPLES)
        worst_z = max(worst_z, abs(closed - estimate) / stderr)
        worst_rel = max(worst_rel, abs(closed - estimate) / closed)
        print(f"  KL closed form {closed:8.4f} | Monte Carlo ({MC_SAMPLES} samples) {estimate:8.4f} ± {stderr:.4f}")
    print(f"closed-form KL vs Monte Carlo on 5 validation images: largest gap {worst_z:.2f} standard errors "
          f"({worst_rel:.2%} of the KL)")

    # --- 3. reparameterization gradients ---------------------------------------------------------
    mu_r = mu.clone().requires_grad_(True)
    logvar_r = logvar.clone().requires_grad_(True)
    eps = torch.randn(mu.shape, generator=torch.Generator().manual_seed(1))
    z = reparameterize(mu_r, logvar_r, torch.Generator().manual_seed(1))  # the same eps
    z.sum().backward()
    reparam_error = max((mu_r.grad - 1).abs().max().item(),
                        (logvar_r.grad - 0.5 * torch.exp(0.5 * logvar) * eps).abs().max().item())
    print(f"reparameterization: largest |autograd - (1, sigma * eps / 2)| = {reparam_error:.1e}")
    return {"bce_value_rel_error": value_error, "bce_grad_rel_error_regular": grad_error,
            "builtin_grad_formula_error": builtin_error, "n_pixels": n_pixels, "n_floored": n_floored,
            "n_floored_ink": n_floored_ink,
            "bce_saturated": [edge_ours, edge_ref], "bce_saturated_grad_finite": edge_grad_finite,
            "bce_saturated_grad_error": edge_grad_error, "bce_saturated_grad_max": edge_grad_max,
            "sum_over_mean_ratio": ratio,
            "kl_mc_worst_stderr": worst_z, "kl_mc_worst_rel": worst_rel, "reparam_grad_error": reparam_error}

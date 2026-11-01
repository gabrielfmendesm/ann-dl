"""Exercise 1 — a VAE on MNIST.

Trains the model of the statement with d = 16 (Figure 1), reports its test terms, draws the reconstructions
and the prior samples (Figures 2 and 3), and measures the numbers the analysis of item E rests on.
"""

import math

import matplotlib.pyplot as plt
import numpy as np
import torch

from common import (BATCH_SIZE, BETA, EPOCHS, LR, SEED, TEST_SEED, VAL_SEED, grey_fraction, save, show_grid,
                    tile_ticks)
from vae import VAE, evaluate, kl_term, reconstruction_term, sample, train

D = 16
PRIOR_SEED = 0   # Figure 3: the prior samples are drawn after torch.manual_seed(0)
GRID = 8         # Figure 3 is an 8 x 8 grid


@torch.no_grad()
def log_aggregate_posterior(Z, mu, logvar, chunk=32):
    """log q(z) for each row of Z, where q(z) = (1/N) sum_n N(z; mu_n, diag(sigma_n^2)) is the aggregate posterior:
    the density of the cloud formed by the codes of all N training images together."""
    n, d = mu.shape
    inv_var = torch.exp(-logvar)
    log_norm = -0.5 * (logvar.sum(dim=1) + d * math.log(2 * math.pi))     # log of each Gaussian's normalizer
    out = []
    for start in range(0, len(Z), chunk):
        z = Z[start:start + chunk]
        sq = ((z[:, None, :] - mu[None]) ** 2 * inv_var[None]).sum(dim=-1)  # (chunk, N) squared Mahalanobis
        out.append(torch.logsumexp(log_norm[None] - 0.5 * sq, dim=1) - math.log(n))
    return torch.cat(out)


@torch.no_grad()
def pixel_entropy(X):
    """Per image, sum_j H(x_j) = -sum_j [x_j log x_j + (1 - x_j) log(1 - x_j)]: the smallest value the BCE can take
    on that image (reached when x_hat = x). It is 0 for black and white pixels and positive for grey ones."""
    return reconstruction_term(X, X).item()


def run(data):
    print("\n" + "=" * 72 + "\nEXERCISE 1 — a VAE on MNIST (d = 16)\n" + "=" * 72)
    X_tr, _ = data["train"]
    X_va, _ = data["val"]
    X_te, y_te = data["test"]

    # --- A: the data -----------------------------------------------------------------------------
    print(f"train {tuple(X_tr.shape)}, validation {tuple(X_va.shape)}, test {tuple(X_te.shape)}; "
          f"pixel range [{X_tr.min().item():.1f}, {X_tr.max().item():.1f}]")

    # --- B: the model ----------------------------------------------------------------------------
    torch.manual_seed(SEED)
    model = VAE(D)
    n_parameters = sum(p.numel() for p in model.parameters())
    print(f"VAE with d = {D}: {n_parameters} parameters")

    # --- C: training, both terms logged separately -----------------------------------------------
    history = train(model, X_tr, X_va, EPOCHS, BATCH_SIZE, LR, BETA, VAL_SEED)
    print(f"untrained model, validation: reconstruction {history['val_rec'][0]:.2f}, KL {history['val_kl'][0]:.4f}")
    print("epoch | train rec   train KL | val rec   val KL   val total")
    for e in range(EPOCHS):
        print(f"{e + 1:5d} | {history['train_rec'][e]:9.2f} {history['train_kl'][e]:9.2f} | "
              f"{history['val_rec'][e + 1]:7.2f} {history['val_kl'][e + 1]:7.2f} "
              f"{history['val_rec'][e + 1] + history['val_kl'][e + 1]:9.2f}")
    test_rec, test_kl = evaluate(model, X_te, TEST_SEED)
    print(f"TEST (d = {D}): reconstruction {test_rec:.2f}, KL {test_kl:.2f}, negative ELBO {test_rec + test_kl:.2f} "
          f"nats per image")

    # Figure 1: the two terms per epoch, training and validation
    epochs = np.arange(1, EPOCHS + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, term, name in ((axes[0], "rec", "reconstruction (BCE)"), (axes[1], "kl", "KL to the prior")):
        ax.plot(epochs, history[f"train_{term}"], "o-", ms=3, color="tab:blue",
                label="training (average over the epoch)")
        ax.plot(epochs, history[f"val_{term}"][1:], "s-", ms=3, color="tab:orange",
                label="validation (end of the epoch)")
        ax.set(title=f"{name} per epoch", xlabel="epoch", ylabel=f"{name}, nats per image", xticks=epochs[1::2])
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle(f"Figure 1 — the two loss terms during training (d = {D}, β = {BETA:g}, Adam lr {LR:g}, "
                 f"batch {BATCH_SIZE})")
    save(fig, "fig1.png")

    # --- D: reconstructions (decoding mu) and samples from the prior -----------------------------
    model.eval()
    first = [int(torch.nonzero(y_te == k)[0]) for k in range(10)]  # first test image of each digit 0-9
    with torch.no_grad():
        mu_first, _ = model.encode(X_te[first])
        recon_first = model.decode(mu_first)
    fig, ax = plt.subplots(figsize=(10, 2.6))
    show_grid(ax, torch.cat([X_te[first], recon_first]), 2, 10)
    tile_ticks(ax, "x", [f"{k}\n#{i}" for k, i in enumerate(first)])
    tile_ticks(ax, "y", ["original", "decoded μ"])
    ax.set(title="Figure 2 — the first test image of each digit (top) and its reconstruction from μ (bottom)",
           xlabel="digit and test index", ylabel="row")
    save(fig, "fig2.png")
    print(f"Figure 2: first test index of each digit 0-9: {first}")

    torch.manual_seed(PRIOR_SEED)
    with torch.no_grad():
        samples = sample(model, GRID * GRID)
    fig, ax = plt.subplots(figsize=(6.2, 6.6))
    show_grid(ax, samples, GRID, GRID)
    tile_ticks(ax, "x", range(1, GRID + 1))
    tile_ticks(ax, "y", range(1, GRID + 1))
    ax.set(title=f"Figure 3 — {GRID * GRID} samples: decoder(z), z ~ N(0, I),\ndrawn after "
                 f"torch.manual_seed({PRIOR_SEED})", xlabel="column", ylabel="row")
    save(fig, "fig3.png")

    # --- E: the numbers behind the analysis ------------------------------------------------------
    with torch.no_grad():
        mu_tr, logvar_tr = model.encode(X_tr)
        mu_te, logvar_te = model.encode(X_te)
        recon_te = model.decode(mu_te)
        rec_from_mu = reconstruction_term(recon_te, X_te).item()
        kl_train = kl_term(mu_tr, logvar_tr).item()
        kl_per_dim = (0.5 * (mu_te ** 2 + logvar_te.exp() - 1 - logvar_te)).mean(dim=0)
        sigma_per_dim = torch.exp(0.5 * logvar_te).mean(dim=0)
        # a decoder that ignores z outputs one image for every input; on the test set the best such image is
        # the per-pixel mean of the test images (it minimizes the summed BCE), so no code-free model does better
        mean_image = X_te.mean(dim=0, keepdim=True).expand_as(X_te)
        rec_no_code = reconstruction_term(mean_image, X_te).item()
        # where the prior samples land: the density of the cloud of training codes, at the codes of the
        # test images (sampled from q(z|x), as in training) and at the 64 codes of Figure 3
        g = torch.Generator().manual_seed(TEST_SEED)
        z_te = mu_te + torch.exp(0.5 * logvar_te) * torch.randn(mu_te.shape, generator=g)
        torch.manual_seed(PRIOR_SEED)
        z_prior = torch.randn(GRID * GRID, D)  # the same draw as Figure 3
        assert torch.equal(model.decode(z_prior), samples), "the codes must be the ones of Figure 3"
        logq_te = log_aggregate_posterior(z_te, mu_tr, logvar_tr)
        logq_prior = log_aggregate_posterior(z_prior, mu_tr, logvar_tr)

    val_kl, val_rec = history["val_kl"], history["val_rec"]
    active = kl_per_dim > 0.1
    logq_p01 = float(torch.quantile(logq_te, 0.01))
    low = torch.nonzero(logq_prior < logq_p01).flatten().tolist()
    low_cells = [[k // GRID + 1, k % GRID + 1] for k in low]  # (row, column) in Figure 3, 1-based
    floor = pixel_entropy(X_te)
    info_bound = kl_train - math.log(len(X_tr))
    results = {
        "d": D, "n_parameters": n_parameters, "history": history,
        "test_rec": test_rec, "test_kl": test_kl, "test_neg_elbo": test_rec + test_kl,
        "first_test_index_per_digit": first,
        "val_kl_first": val_kl[1], "val_kl_last": val_kl[-1], "val_kl_untrained": val_kl[0],
        "val_rec_untrained": val_rec[0], "val_rec_first": val_rec[1], "val_rec_last": val_rec[-1],
        "train_kl_end": kl_train, "rec_no_code": rec_no_code, "bce_floor_test": floor,
        "kl_per_dim": kl_per_dim.tolist(), "sigma_per_dim": sigma_per_dim.tolist(),
        "n_active": int(active.sum()), "max_kl_inactive": float(kl_per_dim[~active].max()) if (~active).any() else None,
        "min_kl_active": float(kl_per_dim[active].min()),
        "sigma_active_range": [float(sigma_per_dim[active].min()), float(sigma_per_dim[active].max())],
        "sigma_inactive_range": ([float(sigma_per_dim[~active].min()), float(sigma_per_dim[~active].max())]
                                 if (~active).any() else None),
        "aggregate_kl_lower_bound": info_bound, "log_n_train": math.log(len(X_tr)),
        "logq_test_median": float(logq_te.median()), "logq_test_p01": logq_p01,
        "logq_prior_median": float(logq_prior.median()), "prior_codes_below_test_p01": len(low),
        "prior_cells_below_test_p01": low_cells,
        "grey_test": grey_fraction(X_te), "grey_recon_test": grey_fraction(recon_te),
        "grey_prior_samples": grey_fraction(samples),
        "test_rec_from_mu": rec_from_mu,
    }
    print(f"\n1E — rising KL: validation KL {val_kl[0]:.4f} untrained, {val_kl[1]:.2f} after epoch 1, "
          f"{val_kl[-1]:.2f} after epoch {EPOCHS}; validation reconstruction {val_rec[0]:.2f} -> {val_rec[1]:.2f} -> "
          f"{val_rec[-1]:.2f}")
    print(f"     epoch 1 -> {EPOCHS}: KL {val_kl[-1] - val_kl[1]:+.2f}, reconstruction "
          f"{val_rec[-1] - val_rec[1]:+.2f}, total {val_rec[-1] + val_kl[-1] - val_rec[1] - val_kl[1]:+.2f} nats")
    print(f"     best test reconstruction of a decoder that ignores z (it outputs the mean test image): "
          f"{rec_no_code:.2f}")
    print(f"     KL per latent dimension (test): {[round(v, 3) for v in kl_per_dim.tolist()]}")
    print(f"     {int(active.sum())} dimensions above 0.1 nat (smallest {results['min_kl_active']:.3f}), the others "
          f"at most {results['max_kl_inactive']:.4f}; mean sigma {[round(v, 2) for v in sigma_per_dim.tolist()]}")
    print(f"1E — Figure 3, reason 1: mean training KL {kl_train:.2f} nats, log N = {math.log(len(X_tr)):.2f}: "
          f"KL(q(z) || p(z)) >= {info_bound:.2f} nats")
    print(f"     log q(z) of the test codes: median {results['logq_test_median']:.2f}, 1st percentile {logq_p01:.2f}; "
          f"of the {GRID * GRID} codes of Figure 3: median {results['logq_prior_median']:.2f}, {len(low)} below that "
          f"1st percentile, at (row, column) {low_cells}")
    print(f"1E — Figure 3, reason 2: grey pixels (0.1 < x < 0.9): test images {results['grey_test']:.2%}, "
          f"reconstructions from mu {results['grey_recon_test']:.2%}, Figure 3 samples "
          f"{results['grey_prior_samples']:.2%}; BCE floor of the test images (sum of pixel entropies) {floor:.2f}")
    print(f"1E — why mu: test reconstruction decoding mu {rec_from_mu:.2f} vs sampled z {test_rec:.2f}")
    return results, model

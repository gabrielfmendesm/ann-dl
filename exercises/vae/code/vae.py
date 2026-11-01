"""The VAE of the statement: the model, the reparameterization, both loss terms, the sampling and the training loop.

Autograd computes the gradients (the statement allows it), but every VAE-specific part is written out here:
the reparameterization z = mu + sigma * eps, the reconstruction term (the binary cross-entropy summed over the
784 pixels), the closed-form KL to the prior N(0, I), and the sampling from that prior. Both loss terms are
returned per image, in nats: summed over pixels (or latent dimensions), averaged over the images of the batch.
"""

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

N_PIXELS = 28 * 28
LOG_FLOOR = -100.0  # the floor F.binary_cross_entropy puts on each log; loss_check.py compares the two


class VAE(nn.Module):
    """Encoder 784 -> 512 -> 256 (ReLU after each), heads 256 -> d for mu and log sigma^2,
    decoder d -> 256 -> 512 -> 784 (ReLU after the hidden layers, sigmoid at the output)."""

    def __init__(self, d):
        super().__init__()
        self.d = d
        self.encoder = nn.Sequential(nn.Linear(N_PIXELS, 512), nn.ReLU(), nn.Linear(512, 256), nn.ReLU())
        self.mu_head = nn.Linear(256, d)
        self.logvar_head = nn.Linear(256, d)  # log sigma^2: unconstrained, so sigma^2 = exp(.) is always positive
        self.decoder = nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Linear(256, 512), nn.ReLU(),
                                     nn.Linear(512, N_PIXELS), nn.Sigmoid())

    def encode(self, x):
        """q(z|x) = N(mu, diag(sigma^2)): returns mu and log sigma^2, each of shape (batch, d)."""
        h = self.encoder(x)
        return self.mu_head(h), self.logvar_head(h)

    def decode(self, z):
        """p(x|z): the probability that each of the 784 pixels is "on" (a Bernoulli mean, in (0, 1))."""
        return self.decoder(z)

    def forward(self, x, generator=None):
        mu, logvar = self.encode(x)
        z = reparameterize(mu, logvar, generator)
        return self.decode(z), mu, logvar


def reparameterize(mu, logvar, generator=None):
    """z = mu + sigma * eps with eps ~ N(0, I): the randomness enters as an input, so gradients reach
    mu (dz/dmu = 1) and log sigma^2 (dz/dlogvar = sigma * eps / 2). generator=None draws eps from the
    global RNG (training); evaluation passes its own generator so that it does not move the global one."""
    sigma = torch.exp(0.5 * logvar)
    eps = torch.randn(mu.shape, generator=generator)
    return mu + sigma * eps


def reconstruction_term(x_hat, x):
    """-log p(x|z) for a Bernoulli decoder: -sum_j [x_j log x_hat_j + (1 - x_j) log(1 - x_hat_j)], the binary
    cross-entropy summed over the 784 pixels and averaged over the images. Each log is floored at -100,
    exactly as F.binary_cross_entropy does, so a saturated sigmoid output would cost 100 nats, not infinity."""
    log_p = torch.log(x_hat).clamp(min=LOG_FLOOR)       # log x_hat
    log_1mp = torch.log1p(-x_hat).clamp(min=LOG_FLOOR)  # log(1 - x_hat)
    per_image = -(x * log_p + (1 - x) * log_1mp).sum(dim=1)
    return per_image.mean()


def kl_term(mu, logvar):
    """KL(N(mu, sigma^2) || N(0, I)) in closed form, 1/2 sum_i (mu_i^2 + sigma_i^2 - 1 - log sigma_i^2),
    summed over the d latent dimensions and averaged over the images."""
    per_image = 0.5 * (mu ** 2 + logvar.exp() - 1 - logvar).sum(dim=1)
    return per_image.mean()


def sample(model, n):
    """n images decoded from z ~ N(0, I), drawn from the global RNG (the caller sets torch.manual_seed)."""
    z = torch.randn(n, model.d)
    return model.decode(z)


@torch.no_grad()
def evaluate(model, X, seed, chunk=1000):
    """Both terms on a whole set, in nats per image. z is sampled as in training (one eps per image), from a
    generator of its own, so the estimate is repeatable and does not touch the training RNG."""
    model.eval()
    generator = torch.Generator().manual_seed(seed)
    rec = kl = 0.0
    for start in range(0, len(X), chunk):
        x = X[start:start + chunk]
        x_hat, mu, logvar = model(x, generator)
        rec += reconstruction_term(x_hat, x).item() * len(x)
        kl += kl_term(mu, logvar).item() * len(x)
    return rec / len(X), kl / len(X)


def train(model, X_train, X_val, epochs, batch_size, lr, beta, val_seed):
    """Adam on the negative ELBO, L = reconstruction + beta * KL, one update per mini-batch.

    Logs both terms separately: on the training set, the average over the epoch's batches (weighted by
    batch size); on the validation set, measured at the end of the epoch with the same eps every epoch
    (generator seeded with val_seed). Entry 0 of the validation lists is the untrained model."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(TensorDataset(X_train), batch_size=batch_size, shuffle=True)  # shuffles with the global RNG
    val_rec, val_kl = evaluate(model, X_val, val_seed)
    history = {"train_rec": [], "train_kl": [], "val_rec": [val_rec], "val_kl": [val_kl]}
    for _ in range(epochs):
        model.train()
        rec_sum = kl_sum = 0.0
        for (x,) in loader:
            x_hat, mu, logvar = model(x)
            rec, kl = reconstruction_term(x_hat, x), kl_term(mu, logvar)
            loss = rec + beta * kl
            if not torch.isfinite(loss):
                raise FloatingPointError("the loss is not finite")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            rec_sum += rec.item() * len(x)
            kl_sum += kl.item() * len(x)
        history["train_rec"].append(rec_sum / len(X_train))
        history["train_kl"].append(kl_sum / len(X_train))
        val_rec, val_kl = evaluate(model, X_val, val_seed)
        history["val_rec"].append(val_rec)
        history["val_kl"].append(val_kl)
    return history

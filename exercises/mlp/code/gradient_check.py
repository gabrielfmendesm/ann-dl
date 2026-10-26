"""Check the hand-derived backward pass against central finite differences.

For every loss the MLP supports, a small random network is built and each
analytic gradient entry is compared with (L(p + e) - L(p - e)) / (2e). The
finite differences are used only as a check: training uses the analytic
gradients of mlp.py.
"""

import numpy as np

from common import SEED
from mlp import MLP, BinaryCrossEntropy, CategoricalCrossEntropy, MeanSquaredError

EPS = 1e-6


def max_relative_error(net, X, y):
    Y = net.loss.targets(y)
    grad_W, grad_b = net.backward(net.forward(X), Y)
    worst = 0.0
    for params, grads in ((net.W, grad_W), (net.b, grad_b)):
        for P, G in zip(params, grads):
            for idx in np.ndindex(P.shape):
                saved = P[idx]
                P[idx] = saved + EPS
                up = net.loss.value(net.forward(X), Y)
                P[idx] = saved - EPS
                down = net.loss.value(net.forward(X), Y)
                P[idx] = saved
                numeric = (up - down) / (2 * EPS)
                worst = max(worst, abs(numeric - G[idx]) / max(1e-8, abs(numeric) + abs(G[idx])))
    return worst


def run():
    print("\n" + "=" * 72 + "\nGRADIENT CHECK — analytic backward pass vs. central finite differences\n" + "=" * 72)
    rng = np.random.default_rng(SEED)
    X = rng.normal(size=(20, 3))
    # each case: (name, network, targets); the generator is consumed in this order
    cases = [("binary cross-entropy, sizes [3, 5, 4, 1]",
              MLP([3, 5, 4, 1], BinaryCrossEntropy(), rng), rng.integers(0, 2, 20)),
             ("categorical cross-entropy, sizes [3, 5, 4, 3]",
              MLP([3, 5, 4, 3], CategoricalCrossEntropy(3), rng), rng.integers(0, 3, 20)),
             ("mean squared error, sizes [3, 5, 4, 2]",
              MLP([3, 5, 4, 2], MeanSquaredError(), rng), rng.uniform(-1, 1, (20, 2)))]
    errors = {}
    for name, net, y in cases:
        errors[name] = max_relative_error(net, X, y)
        print(f"  {name}: max relative error {errors[name]:.1e}")
    assert max(errors.values()) < 1e-6, "backward pass disagrees with finite differences"
    return max(errors.values())

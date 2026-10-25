"""Exercise 1 — one step of backpropagation, checked by the MLP code.

The derivation is done by hand in the report. This script runs the same network
through the from-scratch MLP (MeanSquaredError: tanh output, N = 1) with the
weights of the statement, and prints every intermediate value so that each
hand-computed number can be compared with it.
"""

import numpy as np

from common import SEED
from mlp import MLP, MeanSquaredError

X = np.array([[0.5, -0.2]])
Y = np.array([[1.0]])
W1 = np.array([[0.3, -0.1], [0.2, 0.4]])
B1 = np.array([0.1, -0.2])
W2 = np.array([[0.5, -0.3]])
B2 = np.array([0.2])
ETA = 0.3


def fmt(a):
    return np.array2string(np.asarray(a, dtype=float).squeeze(), precision=6, floatmode="fixed", sign="-")


def run():
    print("=" * 72 + "\nEXERCISE 1 — one step of backpropagation\n" + "=" * 72)
    net = MLP([2, 2, 1], MeanSquaredError(), np.random.default_rng(SEED))
    net.W, net.b = [W1.copy(), W2.copy()], [B1.copy(), B2.copy()]   # the statement's values replace the random start

    # A — forward pass
    U = net.forward(X)
    z1 = X @ W1.T + B1
    h1 = net.activations[1]
    y_hat = net.loss.output(U)
    loss = net.loss.value(U, Y)
    print(f"z1 = {fmt(z1)}\nh1 = {fmt(h1)}\nu2 = {fmt(U)}\ny_hat = {fmt(y_hat)}\nL = {loss:.6f}")

    # B — backward pass (the chain, link by link, then the gradients returned by the MLP)
    dL_dyhat = -2.0 * (Y - y_hat)
    dL_du = net.loss.delta(U, Y)
    dL_dh = dL_du @ W2
    dL_dz = dL_dh * (1.0 - h1 ** 2)
    grad_W, grad_b = net.backward(U, Y)
    print(f"\ndL/dy_hat = {fmt(dL_dyhat)}\ndtanh/du = 1 - y_hat^2 = {fmt(1 - y_hat ** 2)}\ndL/du2 = {fmt(dL_du)}")
    print(f"dL/dW2 = {fmt(grad_W[1])}\ndL/db2 = {fmt(grad_b[1])}")
    print(f"dL/dh1 = {fmt(dL_dh)}\n1 - h1^2 = {fmt(1 - h1 ** 2)}\ndL/dz1 = {fmt(dL_dz)}")
    print(f"dL/dW1 =\n{fmt(grad_W[0])}\ndL/db1 = {fmt(grad_b[0])}")
    assert np.allclose(grad_W[0], np.outer(dL_dz, X)) and np.allclose(grad_b[0], dL_dz)

    # C — parameter update and a second forward pass with the updated parameters
    net.step(grad_W, grad_b, ETA)
    U_new = net.forward(X)
    loss_new = net.loss.value(U_new, Y)
    print(f"\nupdated W2 = {fmt(net.W[1])}\nupdated b2 = {fmt(net.b[1])}")
    print(f"updated W1 =\n{fmt(net.W[0])}\nupdated b1 = {fmt(net.b[0])}")
    print(f"second forward pass: y_hat = {fmt(net.loss.output(U_new))}, L = {loss_new:.6f} "
          f"(was {loss:.6f}, change {loss_new - loss:+.6f})")

    return {"z1": z1.ravel(), "h1": h1.ravel(), "u2": float(U[0, 0]), "y_hat": float(y_hat[0, 0]), "loss": loss,
            "W1": net.W[0], "b1": net.b[0], "W2": net.W[1].ravel(), "b2": float(net.b[1][0]), "loss_after": loss_new}

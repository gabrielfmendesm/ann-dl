"""Multi-layer perceptron written from scratch with NumPy.

One class serves every exercise. The layer sizes and the loss are arguments;
the forward pass, the backward pass and the parameter update are the same code
whatever the task:

    Exercise 1 — MeanSquaredError (tanh output), one sample, weights given by hand
    Exercise 2 — BinaryCrossEntropy (sigmoid output), layer sizes [2, 16, 1]
    Exercise 3 — CategoricalCrossEntropy (softmax output), layer sizes [4, 16, 3]
    Exercise 4 — CategoricalCrossEntropy, layer sizes [4, 16, 16, 3]

Each loss turns the output pre-activation u into a prediction and supplies
dL/du, the error that the backward pass propagates. Hidden layers use tanh.
"""

import numpy as np


# --- Losses: each one pairs an output activation with its loss -----------------------------------
# Every method receives the output pre-activation U (one row per sample). The loss is the mean
# over the N samples, so dL/dU carries the factor 1/N.

class BinaryCrossEntropy:
    """Sigmoid output, L = -mean[y log p + (1 - y) log(1 - p)], one output unit, labels in {0, 1}."""

    def targets(self, y):
        return np.asarray(y, dtype=float).reshape(-1, 1)

    def output(self, U):
        return 0.5 * (1.0 + np.tanh(0.5 * U))           # = 1 / (1 + e^-u), without overflow for large |u|

    def value(self, U, Y):
        # log(1 + e^u) - y u is the same loss written from the logit: no log(0) for saturated p
        return float(np.mean(np.logaddexp(0.0, U) - Y * U))

    def delta(self, U, Y):
        return (self.output(U) - Y) / len(Y)            # dL/du = (p - y) / N


class CategoricalCrossEntropy:
    """Softmax output, L = -mean[sum_k y_k log p_k], one output unit per class, integer labels."""

    def __init__(self, n_classes):
        self.n_classes = n_classes

    def targets(self, y):
        return np.eye(self.n_classes)[np.asarray(y)]   # one-hot rows

    def output(self, U):
        E = np.exp(U - U.max(axis=1, keepdims=True))   # shifting u by a constant leaves softmax unchanged
        return E / E.sum(axis=1, keepdims=True)

    def value(self, U, Y):
        # -log p_y = logsumexp(u) - u_y
        log_norm = U.max(axis=1) + np.log(np.exp(U - U.max(axis=1, keepdims=True)).sum(axis=1))
        return float(np.mean(log_norm - np.sum(Y * U, axis=1)))

    def delta(self, U, Y):
        return (self.output(U) - Y) / len(Y)            # dL/du = (p - y) / N, derived in the report


class MeanSquaredError:
    """Tanh output, L = mean[sum_k (y_k - y_hat_k)^2] (Exercise 1)."""

    def targets(self, y):
        return np.asarray(y, dtype=float).reshape(len(np.atleast_1d(y)), -1)

    def output(self, U):
        return np.tanh(U)

    def value(self, U, Y):
        return float(np.mean(np.sum((Y - self.output(U)) ** 2, axis=1)))

    def delta(self, U, Y):
        Y_hat = self.output(U)
        return 2.0 * (Y_hat - Y) * (1.0 - Y_hat ** 2) / len(Y)   # dL/dy_hat * dy_hat/du


# --- The network ---------------------------------------------------------------------------------

class MLP:
    """Fully connected network: tanh hidden layers, the loss's activation on the output layer.

    layer_sizes = [n_inputs, hidden_1, ..., hidden_k, n_outputs]. W[l] has one row per
    receiving neuron (shape: out x in), as in the lecture, so a layer computes
    Z = A W^T + b for a batch A with one sample per row.
    """

    def __init__(self, layer_sizes, loss, rng):
        self.layer_sizes = list(layer_sizes)
        self.loss = loss
        # LeCun-normal initialization: standard deviation 1/sqrt(fan_in) keeps tanh units out of saturation
        self.W = [rng.normal(0.0, 1.0 / np.sqrt(n_in), size=(n_out, n_in))
                  for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:])]
        self.b = [np.zeros(n_out) for n_out in layer_sizes[1:]]

    def n_parameters(self):
        return sum(W.size + b.size for W, b in zip(self.W, self.b))

    def forward(self, X):
        """Return the output pre-activation U; keep every layer's activation for the backward pass."""
        A = np.atleast_2d(np.asarray(X, dtype=float))
        self.activations = [A]                     # activations[l] is the input of layer l (A[0] = the data)
        last = len(self.W) - 1
        for l, (W, b) in enumerate(zip(self.W, self.b)):
            Z = A @ W.T + b
            if l == last:
                return Z
            A = np.tanh(Z)
            self.activations.append(A)

    def backward(self, U, Y):
        """Gradients of the loss with respect to every W[l] and b[l], by the chain rule."""
        D = self.loss.delta(U, Y)                  # dL/dZ of the output layer, one row per sample
        grad_W, grad_b = [None] * len(self.W), [None] * len(self.b)
        for l in reversed(range(len(self.W))):
            A_in = self.activations[l]             # input of layer l
            grad_W[l] = D.T @ A_in                 # dL/dW[l] = sum over samples of delta x input^T
            grad_b[l] = D.sum(axis=0)              # dL/db[l] = sum over samples of delta
            if l > 0:
                # back through W[l], then through tanh: d tanh(z)/dz = 1 - tanh(z)^2 = 1 - A^2
                D = (D @ self.W[l]) * (1.0 - A_in ** 2)
        return grad_W, grad_b

    def step(self, grad_W, grad_b, lr):
        """Gradient descent: p <- p - lr * dL/dp for every weight and bias."""
        for l in range(len(self.W)):
            self.W[l] -= lr * grad_W[l]
            self.b[l] -= lr * grad_b[l]

    def fit(self, X, y, epochs, lr):
        """Full-batch gradient descent: one update per epoch, using every training sample.

        Returns the training loss at epochs 0..epochs: index 0 is the initial network,
        index e the network after e updates.
        """
        Y = self.loss.targets(y)
        history = []
        for _ in range(epochs):
            U = self.forward(X)
            history.append(self.loss.value(U, Y))
            self.step(*self.backward(U, Y), lr)
        history.append(self.loss.value(self.forward(X), Y))
        return np.array(history)

    def predict_proba(self, X):
        return self.loss.output(self.forward(X))

    def predict(self, X):
        P = self.predict_proba(X)
        return (P[:, 0] >= 0.5).astype(int) if P.shape[1] == 1 else P.argmax(axis=1)

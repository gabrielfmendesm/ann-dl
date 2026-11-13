"""The network of this deliverable: an MLP in PyTorch that returns one logit, and its training loop.

The training loop follows the statement's skeleton: one optimizer step per mini-batch, the validation loss after
every epoch, early stopping with a patience, and the best weights restored at the end (kept in memory rather
than in a file).
"""

import copy

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

BATCH_SIZE = 128     # about 6 positives per batch at the 4.87% rate; a batch without any is rare (0.17%)
MAX_EPOCHS = 300     # an upper bound; early stopping decides
PATIENCE = 30        # epochs without a lower validation loss before stopping


class MLP(nn.Module):
    """n_inputs -> hidden[0] -> ... -> hidden[-1] -> 1 logit, ReLU after every hidden layer and optional dropout
    after every ReLU. Hidden layers use He (Kaiming) normal initialization, the variance that keeps ReLU
    activations from shrinking or growing layer after layer; the output layer, which has no ReLU after it, uses
    Glorot (Xavier) normal. Every bias starts at 0."""

    def __init__(self, n_inputs, hidden, dropout=0.0):
        super().__init__()
        layers, width = [], n_inputs
        for units in hidden:
            linear = nn.Linear(width, units)
            nn.init.kaiming_normal_(linear.weight, nonlinearity="relu")
            nn.init.zeros_(linear.bias)
            layers += [linear, nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            width = units
        output = nn.Linear(width, 1)
        nn.init.xavier_normal_(output.weight)
        nn.init.zeros_(output.bias)
        layers.append(output)
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        """The logit of a stroke, one per row: the loss applies the sigmoid itself."""
        return self.layers(x).squeeze(-1)

    def n_parameters(self):
        """The number of trainable weights and biases."""
        return sum(p.numel() for p in self.parameters())


def to_tensor(a):
    """A float32 tensor, the dtype of the network's weights."""
    return torch.as_tensor(np.asarray(a, dtype=np.float32))


def predict_proba(model, X):
    """Probability of a stroke for each row of X (the sigmoid of the logit), with dropout switched off."""
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(to_tensor(X))).numpy()


def train(X_train, y_train, config, seed, X_val=None, y_val=None, epochs=None):
    """Train one network. Returns (model, history, best_epoch).

    config: hidden (list of widths), dropout, lr, weight_decay. AdamW is used throughout; with weight_decay 0 it is
    exactly Adam. The decay applies to the weight matrices only: biases shift the activations rather than scale them,
    so they are left out, as the course recommends. The loss is BCEWithLogitsLoss with pos_weight = negatives /
    positives of y_train, so the few positives weigh as much in the gradient as all the negatives.

    With a validation set: up to MAX_EPOCHS epochs, early stopping on the validation loss with PATIENCE, and the
    weights of the best epoch restored. Without one (the final retraining): exactly `epochs` epochs.
    Both losses in the history are measured after each epoch with dropout off, so they are comparable.
    """
    if X_val is None and not epochs:
        raise ValueError("without a validation set to stop on, the number of epochs must be given")
    torch.manual_seed(seed)   # the initial weights and the dropout masks
    model = MLP(X_train.shape[1], config["hidden"], config["dropout"])
    y_train = np.asarray(y_train)
    pos_weight = torch.tensor((y_train == 0).sum() / (y_train == 1).sum(), dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    weights = [p for name, p in model.named_parameters() if name.endswith("weight")]
    biases = [p for name, p in model.named_parameters() if name.endswith("bias")]
    optimizer = torch.optim.AdamW([{"params": weights, "weight_decay": config["weight_decay"]},
                                   {"params": biases, "weight_decay": 0.0}], lr=config["lr"])
    Xt, yt = to_tensor(X_train), to_tensor(y_train)
    loader = DataLoader(TensorDataset(Xt, yt), batch_size=BATCH_SIZE, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))   # the order of the mini-batches
    if X_val is not None:
        Xv, yv = to_tensor(X_val), to_tensor(y_val)

    history = {"train_loss": [], "val_loss": [], "val_ap": []}
    best_loss, best_state, best_epoch, wait = float("inf"), None, 0, 0
    for epoch in range(1, (epochs or MAX_EPOCHS) + 1):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            history["train_loss"].append(loss_fn(model(Xt), yt).item())
            if X_val is None:
                continue
            val_logits = model(Xv)
            val_loss = loss_fn(val_logits, yv).item()
        history["val_loss"].append(val_loss)
        history["val_ap"].append(average_precision_score(y_val, val_logits.numpy()))
        if val_loss < best_loss:
            best_loss, best_epoch, wait = val_loss, epoch, 0
            best_state = copy.deepcopy(model.state_dict())   # keep the best weights
        else:
            wait += 1
            if wait == PATIENCE:
                break
    if X_val is not None:
        model.load_state_dict(best_state)
    else:
        best_epoch = epochs
    return model, history, best_epoch

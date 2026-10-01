"""
A 2-2-1 (and 2-2-3) neural model for the sensor-disagreement (XOR) task.

a^(l) = W^(l) h^(l-1) + b^(l),   h^(l) = f^(l)(a^(l))

Binary task : 2 inputs -> 2 hidden units -> 1 output logit, BCEWithLogitsLoss.
3-class task: 2 inputs -> 2 hidden units -> 3 output logits, CrossEntropyLoss.

Both share the same hidden layer - only the output head and loss differ,
matching Task 5's instruction to "modify only the output/loss portion".
"""

import torch
import torch.nn as nn

X = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
Y_BINARY = torch.tensor([[0.0], [1.0], [1.0], [0.0]])          # XOR / disagreement warning
Y_3CLASS = torch.tensor([0, 1, 1, 2])                           # 0=both off,1=disagree,2=both on

ACTIVATIONS = {
    "sigmoid": nn.Sigmoid,
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
}


class XORNet(nn.Module):
    """2 -> 2 -> 1 network. Outputs a raw logit (no sigmoid applied)."""

    def __init__(self, activation="sigmoid"):
        super().__init__()
        self.hidden = nn.Linear(2, 2)
        self.act = ACTIVATIONS[activation]()
        self.output = nn.Linear(2, 1)

    def forward(self, x):
        h = self.act(self.hidden(x))
        return self.output(h)


class ThreeClassNet(nn.Module):
    """2 -> 2 -> 3 network (Task 5): same hidden layer, 3 output logits."""

    def __init__(self, activation="sigmoid"):
        super().__init__()
        self.hidden = nn.Linear(2, 2)
        self.act = ACTIVATIONS[activation]()
        self.output = nn.Linear(2, 3)

    def forward(self, x):
        h = self.act(self.hidden(x))
        return self.output(h)


def zero_init_(model):
    """Set every weight and bias tensor in the model to exactly zero."""
    for p in model.parameters():
        nn.init.zeros_(p)


def make_binary_model(activation="sigmoid", seed=None, zero_init=False):
    if seed is not None:
        torch.manual_seed(seed)
    model = XORNet(activation)
    if zero_init:
        zero_init_(model)
    return model


def make_threeclass_model(activation="sigmoid", seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    return ThreeClassNet(activation)


def train_binary(model, epochs=3000, lr=0.1, record_steps=(0, 1, 5, 20), optimizer_name="adam"):
    """
    Full-batch gradient descent on the 4 XOR examples with BCEWithLogitsLoss.
    Returns a dict of everything Task 4 asks us to record.

    Plain SGD with a fixed learning rate turns out to get stuck in the
    well-known flat local minimum of the sigmoid-XOR loss landscape for
    some initial weights (loss plateaus around 0.69 = ln 2, i.e. the
    network just predicts "0.5 regardless of input" and never escapes).
    Adam's per-parameter adaptive step size was the "justified engineering
    setting" change (Task 4 Part A explicitly allows changing the
    optimiser, not the task) that reliably fixes this without touching
    the architecture, loss function, or data.
    """
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = (torch.optim.Adam(model.parameters(), lr=lr) if optimizer_name == "adam"
                 else torch.optim.SGD(model.parameters(), lr=lr))

    initial_loss = loss_fn(model(X), Y_BINARY).item()
    hidden_rows_over_time = []
    grad_norms_at_step = {}
    losses = []

    for step in range(epochs):
        optimizer.zero_grad()
        logits = model(X)
        loss = loss_fn(logits, Y_BINARY)
        loss.backward()

        if step in record_steps:
            grad_norms_at_step[step] = model.hidden.weight.grad.norm().item()
        if step < 10 or step % 500 == 0:
            hidden_rows_over_time.append(
                (step, model.hidden.weight.data.clone().tolist())
            )
        losses.append(loss.item())

        optimizer.step()

    with torch.no_grad():
        final_logits = model(X)
        probs = torch.sigmoid(final_logits)
        preds = (probs >= 0.5).float()

    return {
        "initial_loss": initial_loss,
        "final_loss": losses[-1],
        "losses": losses,
        "probs": probs.squeeze(-1).tolist(),
        "preds": preds.squeeze(-1).tolist(),
        "targets": Y_BINARY.squeeze(-1).tolist(),
        "all_correct": preds.squeeze(-1).tolist() == Y_BINARY.squeeze(-1).tolist(),
        "hidden_rows_over_time": hidden_rows_over_time,
        "grad_norms_at_step": grad_norms_at_step,
        "final_hidden_grad": model.hidden.weight.grad.clone(),
    }


def train_threeclass(model, epochs=3000, lr=0.5):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    for step in range(epochs):
        optimizer.zero_grad()
        logits = model(X)
        loss = loss_fn(logits, Y_3CLASS)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        final_logits = model(X)
        probs = torch.softmax(final_logits, dim=1)
        preds = probs.argmax(dim=1)

    return {
        "final_loss": loss.item(),
        "logits": final_logits,
        "probs": probs,
        "preds": preds.tolist(),
        "targets": Y_3CLASS.tolist(),
        "all_correct": preds.tolist() == Y_3CLASS.tolist(),
    }

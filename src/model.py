import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.data_loader import load_config


def build_model(input_dim: int, cfg: dict = None) -> nn.Module:
    """MLP: input -> hidden_dims (ReLU + dropout) -> 1 logit. Same model for centralized and federated."""
    cfg = cfg or load_config()
    m = cfg["model"]
    layers, prev = [], input_dim
    for h in m["hidden_dims"]:
        layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(m["dropout"])]
        prev = h
    layers.append(nn.Linear(prev, 1))
    return nn.Sequential(*layers)


def get_weights(model: nn.Module) -> list:
    """Model parameters as a list of numpy arrays (the format Flower sends)."""
    return [v.detach().cpu().numpy() for v in model.state_dict().values()]


def set_weights(model: nn.Module, weights: list) -> None:
    """Load a list of numpy arrays (same order as get_weights) into the model."""
    keys = model.state_dict().keys()
    state = {k: torch.tensor(w) for k, w in zip(keys, weights)}
    model.load_state_dict(state, strict=True)


def train(model, X, y, epochs, lr, batch_size, cfg=None, seed=None):
    """Train in place on numpy arrays X, y. Returns the model.
    Per-epoch training loss is stored in model.loss_history."""
    cfg = cfg or load_config()
    if seed is not None:
        torch.manual_seed(seed)
    Xt = torch.as_tensor(X, dtype=torch.float32)
    yt = torch.as_tensor(y, dtype=torch.float32)

    pos_weight = None
    if cfg["training"]["use_class_weights"]:
        n_pos = float(yt.sum())
        if n_pos > 0:
            pos_weight = torch.tensor([(len(yt) - n_pos) / n_pos])
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(TensorDataset(Xt, yt), batch_size=batch_size, shuffle=True)

    model.train()
    history = []
    for _ in range(epochs):
        total = 0.0
        for xb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(model(xb).squeeze(1), yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(xb)
        history.append(total / len(yt))
    model.loss_history = history
    return model


@torch.no_grad()
def predict_proba(model, X) -> np.ndarray:
    """Probability of the positive class for each row (numpy array)."""
    model.eval()
    logits = model(torch.as_tensor(X, dtype=torch.float32)).squeeze(1)
    return torch.sigmoid(logits).numpy()
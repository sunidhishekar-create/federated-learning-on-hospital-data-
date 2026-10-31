"""Flower ClientApp: one client = one hospital. It only ever loads its own data."""
import os
import time
from functools import lru_cache

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from src.data_loader import get_input_dim, load_config, load_hospital_data
# NumPy-only twins of src.metrics (verified identical by docker/check_client_metrics.py), so a
# client process does not load scikit-learn, SciPy and pandas (~100 MB) just for five metrics.
from src.federated.client_metrics import bootstrap_auc_ci, compute_metrics
from src.model import build_model, predict_proba, train

app = ClientApp()

# In the Docker deployment each hospital has one CPU and OMP_NUM_THREADS=1. Make PyTorch follow
# it for both of its thread pools, so a client process starts no extra threads (and their memory).
# Not set in the simulation, which keeps PyTorch's defaults.
if os.environ.get("OMP_NUM_THREADS"):
    torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
    torch.set_num_interop_threads(1)


@lru_cache(maxsize=None)
def _data(hospital_id: int):
    return load_hospital_data(hospital_id)


def _model_from(msg: Message, cfg: dict):
    model = build_model(get_input_dim(), cfg)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    return model


@app.train()
def train_fn(msg: Message, context: Context):
    """Local training on this hospital's training split."""
    cfg = load_config()
    t = cfg["training"]
    h = int(context.node_config["partition-id"])
    config = msg.content["config"]

    model = _model_from(msg, cfg)
    X, y, _, _ = _data(h)

    start = time.perf_counter()
    train(model, X, y, epochs=int(config["local-epochs"]), lr=t["learning_rate"],
          batch_size=t["batch_size"], cfg=cfg,
          seed=int(config["seed"]) + int(config["server-round"]) * 10 + h)
    seconds = time.perf_counter() - start

    metrics = MetricRecord({
        "num-examples": len(y),
        "hospital": h,
        "train_loss": float(model.loss_history[-1]),
        "train_seconds": seconds,
    })
    return Message(content=RecordDict({"arrays": ArrayRecord(model.state_dict()),
                                       "metrics": metrics}), reply_to=msg)


@app.evaluate()
def evaluate_fn(msg: Message, context: Context):
    """Score the global model on this hospital's own validation split."""
    cfg = load_config()
    h = int(context.node_config["partition-id"])
    model = _model_from(msg, cfg)
    _, _, Xv, yv = _data(h)

    p = predict_proba(model, Xv)
    m = compute_metrics(yv, p, cfg["training"]["threshold"])
    lo, hi = bootstrap_auc_ci(yv, p)

    metrics = MetricRecord({
        "num-examples": m.pop("n"), "hospital": h, **m,
        "roc_auc_ci_lo": lo, "roc_auc_ci_hi": hi,
    })
    return Message(content=RecordDict({"metrics": metrics}), reply_to=msg)

"""Flower ServerApp: FedAvg over the five hospitals, with per-round logging.

Per round it records global-test metrics, per-hospital validation metrics (computed by the
clients on their own data), bytes moved and timing. The final result is written to
results/federated_runs/seed_<seed>.json in the same layout as results/centralized.json.
"""
import json
import time

import numpy as np
import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from src.data_loader import ROOT, get_input_dim, load_config, load_global_test
from src.metrics import evaluate, fairness_gaps
from src.model import build_model

app = ServerApp()

# Filled in by src.federated.run when the app is launched with run_simulation(), which has
# no run-config argument. `flwr run` supplies the same keys through context.run_config.
OVERRIDES: dict = {}


def _opt(context: Context, key: str, default=0):
    return OVERRIDES.get(key, context.run_config.get(key, default))


def _records(replies) -> list:
    """Each reply's metrics as a plain dict."""
    return [dict(next(iter(r.metric_records.values()))) for r in replies]


@app.main()
def main(grid: Grid, context: Context) -> None:
    cfg = load_config()
    fed, H = cfg["federated"], cfg["hospitals"]["num_hospitals"]
    seed = int(_opt(context, "seed", cfg["seed"]))
    rounds = int(_opt(context, "num-server-rounds")) or fed["num_rounds"]
    local_epochs = int(_opt(context, "local-epochs")) or fed["local_epochs"]

    # Same seed -> same initial weights as the centralized run.
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build_model(get_input_dim(), cfg)
    arrays = ArrayRecord(model.state_dict())
    model_bytes = arrays.count_bytes()

    Xte, yte = load_global_test()
    ckpt_dir = ROOT / cfg["paths"]["checkpoints_dir"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_log, site_log, global_log = [], [], {}
    clock = {"eval_s": 0.0, "train_end": None}   # eval_s: time spent evaluating, not training

    def train_aggr(records, weight_key):
        clock["train_end"] = time.perf_counter()
        recs = _records(records)
        train_log.append(sorted(recs, key=lambda r: r["hospital"]))
        n = np.array([r[weight_key] for r in recs], dtype=float)
        loss = np.array([r["train_loss"] for r in recs])
        return MetricRecord({"train_loss": float((loss * n).sum() / n.sum())})

    def eval_aggr(records, weight_key):
        clock["eval_s"] += time.perf_counter() - clock["train_end"]   # client-side evaluation
        recs = _records(records)
        site_log.append(sorted(recs, key=lambda r: r["hospital"]))
        n = np.array([r[weight_key] for r in recs], dtype=float)
        auc = np.array([r["roc_auc"] for r in recs])
        return MetricRecord({"roc_auc_weighted": float((auc * n).sum() / n.sum())})

    def global_eval(server_round: int, arrs: ArrayRecord) -> MetricRecord:
        start = time.perf_counter()
        m = build_model(get_input_dim(), cfg)
        m.load_state_dict(arrs.to_torch_state_dict())
        res = evaluate(m, Xte, yte)
        global_log[server_round] = res
        torch.save(m.state_dict(), ckpt_dir / f"federated_seed{seed}_round{server_round}.pt")
        clock["eval_s"] += time.perf_counter() - start                # server-side evaluation
        return MetricRecord({k: float(v) for k, v in res.items()})

    strategy = FedAvg(
        fraction_train=fed["fraction_fit"], fraction_evaluate=1.0,
        min_train_nodes=H, min_evaluate_nodes=H, min_available_nodes=fed["min_available_clients"],
        train_metrics_aggr_fn=train_aggr, evaluate_metrics_aggr_fn=eval_aggr,
    )

    t0 = time.perf_counter()
    strategy.start(
        grid=grid, initial_arrays=arrays, num_rounds=rounds,
        train_config=ConfigRecord({"seed": seed, "local-epochs": local_epochs}),
        evaluate_fn=global_eval,
    )
    total_wall = time.perf_counter() - t0

    # ---- assemble the result (same layout as results/centralized.json)
    per_site = {str(r["hospital"]): {
        **{k: r[k] for k in ["accuracy", "precision", "recall", "f1", "roc_auc"]},
        "n": int(r["num-examples"]), "n_pos": int(r["n_pos"]),
        "roc_auc_ci": [r["roc_auc_ci_lo"], r["roc_auc_ci_hi"]],
    } for r in site_log[-1]}

    client_seconds = [sum(r["train_seconds"] for r in rnd) for rnd in train_log]
    history = [{
        "round": 0, "global_test": global_log[0],
    }] + [{
        "round": i + 1,
        "global_test": global_log[i + 1],
        "per_site_val": {str(r["hospital"]): {k: r[k] for k in ["recall", "roc_auc"]}
                         for r in site_log[i]},
        "train_loss": float(np.average([r["train_loss"] for r in train_log[i]],
                                       weights=[r["num-examples"] for r in train_log[i]])),
        "client_train_seconds_sum": client_seconds[i],
        "client_train_seconds_max": max(r["train_seconds"] for r in train_log[i]),
        "bytes_train": 2 * H * model_bytes,         # model down + update up, every client
        "bytes_eval_down": H * model_bytes,         # global model sent for local evaluation
    } for i in range(rounds)]

    n_train = int(sum(r["num-examples"] for r in train_log[0]))
    result = {
        "seed": seed,
        "model": cfg["model"],
        "rounds": rounds,
        "local_epochs": local_epochs,
        "n_train": n_train,
        # wall clock of the whole run minus the time spent on evaluation; includes Flower overhead
        "train_seconds": round(total_wall - clock["eval_s"], 2),
        "total_wall_seconds": round(total_wall, 2),
        "client_train_seconds_total": round(sum(client_seconds), 2),
        "model_bytes": model_bytes,
        "bytes_per_round": 2 * H * model_bytes,
        "total_bytes": 2 * H * model_bytes * rounds,
        "total_bytes_with_eval": 3 * H * model_bytes * rounds,
        "global_test": global_log[rounds],
        "per_site_val": per_site,
        "fairness_gaps": fairness_gaps(per_site),
        "history": history,
    }

    out = ROOT / cfg["paths"]["results_dir"] / "federated_runs"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"seed_{seed}.json").write_text(json.dumps(result, indent=2))
    g = result["global_test"]
    print(f"\n[federated seed {seed}] ROC-AUC {g['roc_auc']:.4f} recall {g['recall']:.3f} "
          f"| train {result['train_seconds']} s | {result['total_bytes'] / 1e6:.2f} MB")

"""Run the federated experiment in Flower's simulation runtime, one run per seed.

    python -m src.federated.run              # all eval_seeds -> federated_seeds.json
    python -m src.federated.run --seeds 42   # single run     -> federated.json
"""
import argparse
import json

from flwr.simulation import run_simulation

from src.data_loader import ROOT, load_config
from src.federated import server_app
from src.federated.client_app import app as client_app
from src.multiseed import METRICS, _ms


def run_one(seed: int, rounds: int = 0, local_epochs: int = 0, cfg=None) -> dict:
    cfg = cfg or load_config()
    H = cfg["hospitals"]["num_hospitals"]
    server_app.OVERRIDES.clear()
    server_app.OVERRIDES.update({"seed": seed, "num-server-rounds": rounds,
                                 "local-epochs": local_epochs})
    run_simulation(server_app=server_app.app, client_app=client_app, num_supernodes=H,
                   backend_config={"client_resources": {"num_cpus": 1, "num_gpus": 0}})
    path = ROOT / cfg["paths"]["results_dir"] / "federated_runs" / f"seed_{seed}.json"
    return json.loads(path.read_text())


def summarize(runs: list, cfg=None) -> dict:
    """Mean/std over seeds, same structure as results/centralized_seeds.json."""
    H = (cfg or load_config())["hospitals"]["num_hospitals"]
    return {
        "seeds": [r["seed"] for r in runs],
        "rounds": runs[0]["rounds"],
        "local_epochs": runs[0]["local_epochs"],
        "global_test": {m: _ms([r["global_test"][m] for r in runs]) for m in METRICS},
        "train_seconds": _ms([r["train_seconds"] for r in runs]),
        "client_train_seconds_total": _ms([r["client_train_seconds_total"] for r in runs]),
        "total_bytes": _ms([r["total_bytes"] for r in runs]),
        "roc_auc_gap": _ms([r["fairness_gaps"]["roc_auc"] for r in runs]),
        "per_site": {str(h): {
            "roc_auc": _ms([r["per_site_val"][str(h)]["roc_auc"] for r in runs]),
            "recall": _ms([r["per_site_val"][str(h)]["recall"] for r in runs]),
        } for h in range(H)},
        "runs": [{
            "seed": r["seed"], "train_seconds": r["train_seconds"],
            "global_test": {m: r["global_test"][m] for m in METRICS},
            "site_roc_auc": [r["per_site_val"][str(h)]["roc_auc"] for h in range(H)],
            "site_recall": [r["per_site_val"][str(h)]["recall"] for h in range(H)],
            "roc_auc_gap": r["fairness_gaps"]["roc_auc"],
        } for r in runs],
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", help="default: eval_seeds from config")
    ap.add_argument("--rounds", type=int, default=0, help="default: federated.num_rounds")
    ap.add_argument("--local-epochs", type=int, default=0, help="default: federated.local_epochs")
    a = ap.parse_args()

    cfg = load_config()
    seeds = a.seeds or cfg["eval_seeds"]
    res = ROOT / cfg["paths"]["results_dir"]
    runs = [run_one(s, a.rounds, a.local_epochs, cfg) for s in seeds]

    if cfg["seed"] in seeds:                          # the single-run file, like centralized.json
        r = next(r for r in runs if r["seed"] == cfg["seed"])
        (res / "federated.json").write_text(json.dumps(r, indent=2))
    if len(seeds) > 1:
        (res / "federated_seeds.json").write_text(json.dumps(summarize(runs, cfg), indent=2))

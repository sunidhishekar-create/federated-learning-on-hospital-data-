"""Local-only baseline: every hospital trains its own model on its own training split, with
no sharing at all. This is what a hospital gets if it does not join the federation.

Same model, hyperparameters, epoch budget (centralized.epochs = 8) and seeds as the
centralized and federated runs. Each hospital's model is scored on that hospital's own
validation set (the per-site comparison) and on the global test set (how well it carries
over to other patients). Writes results/local_only_seeds.json and one checkpoint per
hospital and seed (results/checkpoints/local_h<h>_seed<s>.pt, used by src/compare.py).

    python -m src.local_only
"""
import json
import time

import numpy as np
import torch

from src.data_loader import (ROOT, get_input_dim, load_config, load_global_test,
                             load_hospital_data)
from src.metrics import evaluate, fairness_gaps
from src.model import build_model, train
from src.multiseed import METRICS, _ms


def run_local(h: int, seed: int, cfg: dict):
    """Train hospital h's own model. Returns (model, result dict)."""
    t = cfg["training"]
    torch.manual_seed(seed)                            # same initial weights as the other runs
    np.random.seed(seed)
    X, y, Xv, yv = load_hospital_data(h)
    model = build_model(get_input_dim(), cfg)
    start = time.perf_counter()
    train(model, X, y, epochs=cfg["centralized"]["epochs"], lr=t["learning_rate"],
          batch_size=t["batch_size"], cfg=cfg, seed=seed)
    seconds = time.perf_counter() - start
    return model, {"n_train": int(len(y)), "train_seconds": round(seconds, 2),
                   "own_val": evaluate(model, Xv, yv)}


def run_all(cfg=None) -> dict:
    cfg = cfg or load_config()
    H, seeds = cfg["hospitals"]["num_hospitals"], cfg["eval_seeds"]
    Xte, yte = load_global_test()
    ckpt = ROOT / cfg["paths"]["checkpoints_dir"]
    ckpt.mkdir(parents=True, exist_ok=True)

    runs = []
    for s in seeds:
        sites = {}
        for h in range(H):
            model, r = run_local(h, s, cfg)
            r["global_test"] = evaluate(model, Xte, yte)
            sites[str(h)] = r
            torch.save(model.state_dict(), ckpt / f"local_h{h}_seed{s}.pt")
        own = {h: v["own_val"] for h, v in sites.items()}
        runs.append({"seed": s, "sites": sites, "roc_auc_gap": fairness_gaps(own)["roc_auc"]})
        print(f"seed {s}: own-validation ROC-AUC "
              + " ".join(f"h{h}={v['own_val']['roc_auc']:.3f}" for h, v in sites.items()))

    hs = [str(h) for h in range(H)]
    summary = {
        "seeds": list(seeds),
        "epochs": cfg["centralized"]["epochs"],
        # same structure as per_site in centralized_seeds.json / federated_seeds.json
        "per_site": {h: {m: _ms([r["sites"][h]["own_val"][m] for r in runs]) for m in ("roc_auc", "recall")}
                     for h in hs},
        "roc_auc_gap": _ms([r["roc_auc_gap"] for r in runs]),
        "global_test_by_hospital_model": {
            h: {m: _ms([r["sites"][h]["global_test"][m] for r in runs]) for m in METRICS} for h in hs},
        "train_seconds": {h: _ms([r["sites"][h]["train_seconds"] for r in runs]) for h in hs},
        "n_train": {h: runs[0]["sites"][h]["n_train"] for h in hs},
        "runs": runs,
    }
    out = ROOT / cfg["paths"]["results_dir"] / "local_only_seeds.json"
    out.write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    s = run_all()
    print(f"\nmean +/- std over {len(s['seeds'])} seeds (own validation set | global test set)")
    for h, v in s["per_site"].items():
        g = s["global_test_by_hospital_model"][h]["roc_auc"]
        print(f"  hospital {h}: ROC-AUC {v['roc_auc']['mean']:.3f} +/- {v['roc_auc']['std']:.3f} "
              f"| recall {v['recall']['mean']:.3f} | global test ROC-AUC {g['mean']:.3f}")
    g = s["roc_auc_gap"]
    print(f"  per-site ROC-AUC gap {g['mean']:.3f} +/- {g['std']:.3f}")

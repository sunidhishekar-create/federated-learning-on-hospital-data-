"""Check that src/federated/client_metrics.py (NumPy only, used by the ClientApp) gives the same
results as src/metrics.py (scikit-learn). Writes results/docker/client_metrics_equivalence.json.

    python docker/check_client_metrics.py

Uses every saved federated checkpoint (results/checkpoints/federated_seed*_round*.pt) on all five
hospital validation sets and the global test set, plus edge cases (ties, one class only, nothing
or everything predicted positive).
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np                                                      # noqa: E402
import torch                                                            # noqa: E402

from src import metrics as sk                                           # noqa: E402
from src.data_loader import (get_input_dim, load_config, load_global_test,   # noqa: E402
                             load_hospital_data)
from src.federated import client_metrics as lean                       # noqa: E402
from src.model import build_model, predict_proba                        # noqa: E402

KEYS = ["accuracy", "precision", "recall", "f1", "roc_auc", "n", "n_pos"]


def compare(y, p, thr, worst):
    a, b = sk.compute_metrics(y, p, thr), lean.compute_metrics(y, p, thr)
    for k in KEYS:
        if np.isnan(a[k]) and np.isnan(b[k]):
            continue
        worst[k] = max(worst.get(k, 0.0), abs(a[k] - b[k]))
    if len(np.unique(y)) == 2:
        ca, cb = sk.bootstrap_auc_ci(y, p), lean.bootstrap_auc_ci(y, p)
        worst["roc_auc_ci"] = max(worst.get("roc_auc_ci", 0.0), abs(ca[0] - cb[0]), abs(ca[1] - cb[1]))


def main():
    cfg = load_config()
    thr = cfg["training"]["threshold"]
    sets = {str(h): load_hospital_data(h)[2:] for h in range(5)}
    sets["global_test"] = load_global_test()
    ckpts = sorted((ROOT / cfg["paths"]["checkpoints_dir"]).glob("federated_seed*_round*.pt"))
    worst, n_cases = {}, 0
    for c in ckpts:
        m = build_model(get_input_dim(), cfg)
        m.load_state_dict(torch.load(c))
        for X, y in sets.values():
            p = predict_proba(m, X)
            compare(y, p, thr, worst)
            compare(y, np.round(p, 2), thr, worst)          # many tied scores
            n_cases += 2
    y, p = sets["0"][1], np.linspace(0, 1, len(sets["0"][1]))
    edge = {"one class only": (np.zeros(50, int), np.random.default_rng(0).random(50)),
            "nothing predicted positive": (y, p * 0.1),
            "everything predicted positive": (y, 0.6 + p * 0.1),
            "all scores equal": (y, np.full(len(y), 0.3))}
    for y_, p_ in edge.values():
        compare(y_, p_, thr, worst)
        n_cases += 1
    out = {"checkpoints": len(ckpts), "cases_compared": n_cases, "edge_cases": list(edge),
           "max_abs_difference": worst,
           "identical_within_1e-12": all(v <= 1e-12 for v in worst.values())}
    path = ROOT / "results" / "docker" / "client_metrics_equivalence.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

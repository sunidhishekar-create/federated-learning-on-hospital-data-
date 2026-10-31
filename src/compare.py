"""Paired comparison of federated vs centralized models -> results/comparison.json.

Needs per-patient predictions from both models, so it (a) retrains the centralized model for
each seed (same code path as src/centralized.py) and (b) loads the final federated checkpoint
for each seed (results/checkpoints/federated_seed<S>_round<R>.pt).

Paired bootstrap: each draw resamples the rows of a validation set (or the global test set)
ONCE and scores both models on those same rows, so the difference isolates the model, not
the sample. Differences are averaged over seeds inside every draw, so the interval covers
row-sampling noise on top of the seed average. Hospitals are resampled independently
(they hold different patients).

    python -m src.compare
"""
import json

import numpy as np
import torch
from scipy.stats import rankdata

from src.centralized import load_combined_train
from src.data_loader import (ROOT, get_input_dim, load_config, load_global_test,
                             load_hospital_data)
from src.model import build_model, predict_proba, train

B = 1000          # bootstrap draws
BOOT_SEED = 2024


def fast_auc(y, p) -> float:
    """ROC-AUC via ranks (same value as sklearn, much faster inside a bootstrap loop)."""
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(p)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def recall_at(y, p, thr) -> float:
    pos = y == 1
    return float((p[pos] >= thr).mean()) if pos.any() else float("nan")


def centralized_model(seed: int, cfg: dict):
    t, H = cfg["training"], cfg["hospitals"]["num_hospitals"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    X, y = load_combined_train(H)
    model = build_model(get_input_dim(), cfg)
    train(model, X, y, epochs=cfg["centralized"]["epochs"], lr=t["learning_rate"],
          batch_size=t["batch_size"], cfg=cfg, seed=seed)
    return model


def federated_model(seed: int, rounds: int, cfg: dict):
    model = build_model(get_input_dim(), cfg)
    ckpt = ROOT / cfg["paths"]["checkpoints_dir"] / f"federated_seed{seed}_round{rounds}.pt"
    model.load_state_dict(torch.load(ckpt))
    return model


def main():
    cfg = load_config()
    H, seeds = cfg["hospitals"]["num_hospitals"], cfg["eval_seeds"]
    thr = cfg["training"]["threshold"]
    rounds = cfg["federated"]["num_rounds"]

    # datasets: "global" (test set) and one validation set per hospital
    Xte, yte = load_global_test()
    data = {"global": (Xte, yte)}
    for h in range(H):
        _, _, Xv, yv = load_hospital_data(h)
        data[str(h)] = (Xv, yv)
    names = list(data)

    # per-patient probabilities: probs[model][seed][dataset]
    probs = {"cen": {}, "fed": {}}
    for s in seeds:
        for key, model in (("cen", centralized_model(s, cfg)),
                           ("fed", federated_model(s, rounds, cfg))):
            probs[key][s] = {d: predict_proba(model, data[d][0]) for d in names}
        print(f"seed {s}: predictions ready")

    # cross-check against the numbers already recorded
    check = {}
    for key, path in (("cen", "centralized_seeds_mac.json"), ("fed", "federated_seeds.json")):
        rec = json.loads((ROOT / cfg["paths"]["results_dir"] / path).read_text())["runs"]
        ours = [fast_auc(yte, probs[key][s]["global"]) for s in seeds]
        theirs = [r["global_test"]["roc_auc"] for r in rec]
        check[key] = {"recomputed": ours, "recorded": theirs,
                      "max_abs_diff": float(np.max(np.abs(np.array(ours) - np.array(theirs))))}
        print(f"{key}: max |recomputed - recorded| ROC-AUC = {check[key]['max_abs_diff']:.2e}")

    # point estimates on the original data (mean over seeds)
    def mean_over_seeds(key, d, fn):
        y = data[d][1]
        return float(np.mean([fn(y, probs[key][s][d]) for s in seeds]))
    point = {d: {key: {"roc_auc": mean_over_seeds(key, d, fast_auc),
                       "recall": mean_over_seeds(key, d, lambda y, p: recall_at(y, p, thr))}
                 for key in ("cen", "fed")} for d in names}

    # paired bootstrap
    rng = np.random.default_rng(BOOT_SEED)
    boot = {d: {k: np.empty(B) for k in ("auc_cen", "auc_fed", "rec_cen", "rec_fed")}
            for d in names}
    for b in range(B):
        for d in names:
            y = data[d][1]
            idx = rng.integers(0, len(y), len(y))
            yb = y[idx]
            for key in ("cen", "fed"):
                boot[d][f"auc_{key}"][b] = np.mean(
                    [fast_auc(yb, probs[key][s][d][idx]) for s in seeds])
                boot[d][f"rec_{key}"][b] = np.mean(
                    [recall_at(yb, probs[key][s][d][idx], thr) for s in seeds])
        if (b + 1) % 250 == 0:
            print(f"bootstrap {b + 1}/{B}")

    def summarize(delta, estimate):
        lo, hi = np.percentile(delta, [2.5, 97.5])
        p = min(1.0, 2 * min((delta <= 0).mean(), (delta >= 0).mean()))
        return {"diff": float(estimate), "ci_lo": float(lo), "ci_hi": float(hi),
                "p": float(max(p, 1.0 / B))}

    diffs = {}
    for d in names:
        diffs[d] = {
            "roc_auc": summarize(boot[d]["auc_fed"] - boot[d]["auc_cen"],
                                 point[d]["fed"]["roc_auc"] - point[d]["cen"]["roc_auc"]),
            "recall": summarize(boot[d]["rec_fed"] - boot[d]["rec_cen"],
                                point[d]["fed"]["recall"] - point[d]["cen"]["recall"]),
        }

    # fairness summaries (computed on seed-averaged AUC per hospital)
    sites = [str(h) for h in range(H)]
    def stack(key):
        return np.stack([boot[h][f"auc_{key}"] for h in sites])           # (H, B)
    ac, af = stack("cen"), stack("fed")
    gap_c, gap_f = ac.max(0) - ac.min(0), af.max(0) - af.min(0)
    worst_c, worst_f = ac.min(0), af.min(0)
    pc = np.array([point[h]["cen"]["roc_auc"] for h in sites])
    pf = np.array([point[h]["fed"]["roc_auc"] for h in sites])
    fairness = {
        "gap_cen": float(pc.max() - pc.min()), "gap_fed": float(pf.max() - pf.min()),
        "gap_diff": summarize(gap_f - gap_c, (pf.max() - pf.min()) - (pc.max() - pc.min())),
        "worst_site_diff": summarize(worst_f - worst_c, pf.min() - pc.min()),
        "worst_site_cen": float(pc.min()), "worst_site_fed": float(pf.min()),
    }

    out = {"bootstrap_draws": B, "seeds": seeds, "rounds": rounds, "check": check,
           "point": point, "diffs": diffs, "fairness": fairness}
    path = ROOT / cfg["paths"]["results_dir"] / "comparison.json"
    path.write_text(json.dumps(out, indent=2))
    print("wrote", path)


if __name__ == "__main__":
    main()

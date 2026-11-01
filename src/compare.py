"""Paired comparison of federated, centralized and local-only models -> results/comparison.json.

Needs per-patient predictions from every model, so it (a) retrains the centralized model for
each seed (same code path as src/centralized.py), (b) loads the final federated checkpoint
for each seed (results/checkpoints/federated_seed<S>_round<R>.pt) and (c) loads each hospital's
local-only model (results/checkpoints/local_h<H>_seed<S>.pt, from src/local_only.py).

Paired bootstrap: each draw resamples the rows of a validation set (or the global test set)
ONCE and scores all models on those same rows, so a difference isolates the model, not
the sample. Differences are averaged over seeds inside every draw, so the interval covers
row-sampling noise on top of the seed average. Hospitals are resampled independently
(they hold different patients).

Besides ROC-AUC and recall, per-hospital results include an AGE-ADJUSTED ROC-AUC: it only
compares patients within the same age band (<50, 50-69, 70+). The simulated hospitals were
built by assigning age band and label separately, which makes age a hospital-specific
shortcut for the label (most extreme in hospital 3, where almost no 50-69 year olds are
readmitted). The adjusted AUC cannot be won by learning that shortcut.

    python -m src.compare        # after src.multiseed, src.federated.run and src.local_only
"""
import json

import joblib
import numpy as np
import torch
from scipy.stats import rankdata

from src.centralized import load_combined_train
from src.data_loader import (ROOT, _processed_dir, get_input_dim, load_config,
                             load_global_test, load_hospital_data)
from src.model import build_model, predict_proba, train

B = 1000          # bootstrap draws
BOOT_SEED = 2024
AGE_BANDS = (50, 70)   # band edges in years: <50 | 50-69 | 70+ (same bands as src/partition.py)


def fast_auc(y, p) -> float:
    """ROC-AUC via ranks (same value as sklearn, much faster inside a bootstrap loop)."""
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(p)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def auc_within(y, p, band) -> float:
    """ROC-AUC over (positive, negative) pairs from the SAME age band only."""
    num = den = 0.0
    for b in np.unique(band):
        m = band == b
        n1 = int(y[m].sum())
        n0 = int(m.sum()) - n1
        if n1 and n0:
            num += fast_auc(y[m], p[m]) * n1 * n0
            den += n1 * n0
    return num / den if den else float("nan")


def recall_at(y, p, thr) -> float:
    pos = y == 1
    return float((p[pos] >= thr).mean()) if pos.any() else float("nan")


def age_band_fn():
    """Map encoded rows back to age bands, using the scaler fitted in src.data_loader."""
    meta = json.loads((_processed_dir() / "meta.json").read_text())
    col = meta["feature_names"].index("num__age")
    pre = joblib.load(_processed_dir() / "preprocessor.joblib")
    scaler, i = pre.named_transformers_["num"], meta["num_cols"].index("age")
    return lambda X: np.digitize(X[:, col] * scaler.scale_[i] + scaler.mean_[i], AGE_BANDS)


def centralized_model(seed: int, cfg: dict):
    t, H = cfg["training"], cfg["hospitals"]["num_hospitals"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    X, y = load_combined_train(H)
    model = build_model(get_input_dim(), cfg)
    train(model, X, y, epochs=cfg["centralized"]["epochs"], lr=t["learning_rate"],
          batch_size=t["batch_size"], cfg=cfg, seed=seed)
    return model


def load_checkpoint(name: str, cfg: dict):
    model = build_model(get_input_dim(), cfg)
    model.load_state_dict(torch.load(ROOT / cfg["paths"]["checkpoints_dir"] / name))
    return model


def main():
    cfg = load_config()
    H, seeds = cfg["hospitals"]["num_hospitals"], cfg["eval_seeds"]
    thr = cfg["training"]["threshold"]
    rounds = cfg["federated"]["num_rounds"]
    res = ROOT / cfg["paths"]["results_dir"]
    band_of = age_band_fn()

    # datasets: "global" (test set) and one validation set per hospital
    Xte, yte = load_global_test()
    data = {"global": (Xte, yte)}
    for h in range(H):
        _, _, Xv, yv = load_hospital_data(h)
        data[str(h)] = (Xv, yv)
    names = list(data)
    sites = [str(h) for h in range(H)]
    bands = {d: band_of(data[d][0]) for d in sites}

    # per-patient probabilities: probs[model][seed][dataset]. "loc" = the hospital's own local
    # model on its own validation set; "loc_global" = each hospital's local model on the test set.
    probs = {"cen": {}, "fed": {}, "loc": {}, "loc_global": {}}
    for s in seeds:
        for key, model in (("cen", centralized_model(s, cfg)),
                           ("fed", load_checkpoint(f"federated_seed{s}_round{rounds}.pt", cfg))):
            probs[key][s] = {d: predict_proba(model, data[d][0]) for d in names}
        locals_ = {h: load_checkpoint(f"local_h{h}_seed{s}.pt", cfg) for h in sites}
        probs["loc"][s] = {h: predict_proba(locals_[h], data[h][0]) for h in sites}
        probs["loc_global"][s] = {h: predict_proba(locals_[h], Xte) for h in sites}
        print(f"seed {s}: predictions ready")

    # cross-check against the numbers already recorded
    check = {}
    for key, path in (("cen", "centralized_seeds_mac.json"), ("fed", "federated_seeds.json")):
        rec = json.loads((res / path).read_text())["runs"]
        ours = [fast_auc(yte, probs[key][s]["global"]) for s in seeds]
        theirs = [r["global_test"]["roc_auc"] for r in rec]
        check[key] = {"recomputed": ours, "recorded": theirs,
                      "max_abs_diff": float(np.max(np.abs(np.array(ours) - np.array(theirs))))}
    rec = json.loads((res / "local_only_seeds.json").read_text())["runs"]
    ours = [fast_auc(data[h][1], probs["loc"][s][h]) for s in seeds for h in sites]
    theirs = [r["sites"][h]["own_val"]["roc_auc"] for r in rec for h in sites]
    check["loc"] = {"max_abs_diff": float(np.max(np.abs(np.array(ours) - np.array(theirs))))}
    for key, v in check.items():
        print(f"{key}: max |recomputed - recorded| ROC-AUC = {v['max_abs_diff']:.2e}")

    # point estimates on the original data (mean over seeds)
    def mean_over_seeds(key, d, fn):
        y = data[d][1]
        return float(np.mean([fn(y, probs[key][s][d]) for s in seeds]))
    point = {d: {key: {"roc_auc": mean_over_seeds(key, d, fast_auc),
                       "recall": mean_over_seeds(key, d, lambda y, p: recall_at(y, p, thr))}
                 for key in ("cen", "fed")} for d in names}
    for h in sites:
        point[h]["loc"] = {"roc_auc": mean_over_seeds("loc", h, fast_auc),
                           "recall": mean_over_seeds("loc", h, lambda y, p: recall_at(y, p, thr))}
        for key in ("cen", "fed", "loc"):
            point[h][key]["roc_auc_within_age"] = mean_over_seeds(
                key, h, lambda y, p, b=bands[h]: auc_within(y, p, b))
    point_loc_global = {h: float(np.mean([fast_auc(yte, probs["loc_global"][s][h]) for s in seeds]))
                        for h in sites}

    # paired bootstrap
    rng = np.random.default_rng(BOOT_SEED)
    boot = {d: {k: np.empty(B) for k in ("auc_cen", "auc_fed", "rec_cen", "rec_fed")}
            for d in names}
    for h in sites:
        boot[h].update({k: np.empty(B) for k in ("auc_loc", "rec_loc", "aw_cen", "aw_fed", "aw_loc")})
    boot_loc_global = {h: np.empty(B) for h in sites}
    for b in range(B):
        for d in names:
            y = data[d][1]
            idx = rng.integers(0, len(y), len(y))
            yb = y[idx]
            keys = ("cen", "fed") if d == "global" else ("cen", "fed", "loc")
            for key in keys:
                boot[d][f"auc_{key}"][b] = np.mean(
                    [fast_auc(yb, probs[key][s][d][idx]) for s in seeds])
                boot[d][f"rec_{key}"][b] = np.mean(
                    [recall_at(yb, probs[key][s][d][idx], thr) for s in seeds])
                if d != "global":
                    boot[d][f"aw_{key}"][b] = np.mean(
                        [auc_within(yb, probs[key][s][d][idx], bands[d][idx]) for s in seeds])
            if d == "global":
                for h in sites:
                    boot_loc_global[h][b] = np.mean(
                        [fast_auc(yb, probs["loc_global"][s][h][idx]) for s in seeds])
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

    # each hospital: what joining buys over training alone (positive = better than local-only)
    def pair(h, a, b):
        return {
            "roc_auc": summarize(boot[h][f"auc_{a}"] - boot[h][f"auc_{b}"],
                                 point[h][a]["roc_auc"] - point[h][b]["roc_auc"]),
            "recall": summarize(boot[h][f"rec_{a}"] - boot[h][f"rec_{b}"],
                                point[h][a]["recall"] - point[h][b]["recall"]),
            "roc_auc_within_age": summarize(boot[h][f"aw_{a}"] - boot[h][f"aw_{b}"],
                                            point[h][a]["roc_auc_within_age"] - point[h][b]["roc_auc_within_age"]),
        }
    vs_local = {h: {"fed_minus_local": pair(h, "fed", "loc"), "cen_minus_local": pair(h, "cen", "loc"),
                    "fed_minus_cen": pair(h, "fed", "cen")} for h in sites}
    global_local = {h: {"auc_local_model": point_loc_global[h],
                        "fed_minus_local": summarize(boot["global"]["auc_fed"] - boot_loc_global[h],
                                                     point["global"]["fed"]["roc_auc"] - point_loc_global[h])}
                    for h in sites}

    # fairness summaries (computed on seed-averaged AUC per hospital)
    def stack(key, prefix="auc"):
        return np.stack([boot[h][f"{prefix}_{key}"] for h in sites])      # (H, B)
    ac, af = stack("cen"), stack("fed")
    gap_c, gap_f = ac.max(0) - ac.min(0), af.max(0) - af.min(0)
    worst_c, worst_f = ac.min(0), af.min(0)
    pc = np.array([point[h]["cen"]["roc_auc"] for h in sites])
    pf = np.array([point[h]["fed"]["roc_auc"] for h in sites])
    pl = np.array([point[h]["loc"]["roc_auc"] for h in sites])
    fairness = {
        "gap_cen": float(pc.max() - pc.min()), "gap_fed": float(pf.max() - pf.min()),
        "gap_diff": summarize(gap_f - gap_c, (pf.max() - pf.min()) - (pc.max() - pc.min())),
        "worst_site_diff": summarize(worst_f - worst_c, pf.min() - pc.min()),
        "worst_site_cen": float(pc.min()), "worst_site_fed": float(pf.min()),
        "gap_loc": float(pl.max() - pl.min()), "worst_site_loc": float(pl.min()),
        "within_age": {k: {"gap": float(v.max() - v.min()), "worst_site": float(v.min())} for k, v in
                       {key: np.array([point[h][key]["roc_auc_within_age"] for h in sites])
                        for key in ("cen", "fed", "loc")}.items()},
    }

    out = {"bootstrap_draws": B, "seeds": seeds, "rounds": rounds, "check": check,
           "point": point, "diffs": diffs, "fairness": fairness,
           "vs_local": vs_local, "global_test_local_models": global_local}
    path = res / "comparison.json"
    path.write_text(json.dumps(out, indent=2))
    print("wrote", path)


if __name__ == "__main__":
    main()

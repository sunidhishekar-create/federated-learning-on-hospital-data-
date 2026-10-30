import json

import numpy as np

from src.centralized import run_centralized
from src.data_loader import ROOT, load_config

METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc"]


def _ms(values) -> dict:
    """Mean and sample standard deviation (n-1) of a list of numbers."""
    v = np.asarray(values, dtype=float)
    return {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0}


def run_multiseed(cfg=None, seeds=None) -> dict:
    """Run the centralized baseline once per seed. Data and hospitals stay fixed (they are
    read from data/processed); only the model's initial weights and batch order change."""
    cfg = cfg or load_config()
    seeds = seeds or cfg["eval_seeds"]
    H = cfg["hospitals"]["num_hospitals"]

    runs = []
    for s in seeds:
        c = dict(cfg)
        c["seed"] = s
        r = run_centralized(c, save=False)
        runs.append({
            "seed": s,
            "train_seconds": r["train_seconds"],
            "global_test": {m: r["global_test"][m] for m in METRICS},
            "site_roc_auc": [r["per_site_val"][h]["roc_auc"] for h in range(H)],
            "site_recall": [r["per_site_val"][h]["recall"] for h in range(H)],
            "roc_auc_gap": r["fairness_gaps"]["roc_auc"],
        })
        print(f"seed {s}: ROC-AUC {r['global_test']['roc_auc']:.4f}, "
              f"recall {r['global_test']['recall']:.3f}, {r['train_seconds']} s")

    summary = {
        "seeds": list(seeds),
        "epochs": cfg["centralized"]["epochs"],
        "global_test": {m: _ms([x["global_test"][m] for x in runs]) for m in METRICS},
        "train_seconds": _ms([x["train_seconds"] for x in runs]),
        "roc_auc_gap": _ms([x["roc_auc_gap"] for x in runs]),
        "per_site": {
            str(h): {
                "roc_auc": _ms([x["site_roc_auc"][h] for x in runs]),
                "recall": _ms([x["site_recall"][h] for x in runs]),
            } for h in range(H)
        },
        "runs": runs,
    }

    out = ROOT / cfg["paths"]["results_dir"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "centralized_seeds.json").write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    s = run_multiseed()
    print("\nmean +/- std over", len(s["seeds"]), "seeds")
    for m, v in s["global_test"].items():
        print(f"  {m:10s} {v['mean']:.4f} +/- {v['std']:.4f}")
    t = s["train_seconds"]
    print(f"  train time {t['mean']:.1f} +/- {t['std']:.1f} s")
    g = s["roc_auc_gap"]
    print(f"  per-site ROC-AUC gap {g['mean']:.3f} +/- {g['std']:.3f}")
    for h, v in s["per_site"].items():
        print(f"  hospital {h}: ROC-AUC {v['roc_auc']['mean']:.3f} +/- {v['roc_auc']['std']:.3f}"
              f" | recall {v['recall']['mean']:.3f} +/- {v['recall']['std']:.3f}")
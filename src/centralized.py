import json
import time

import numpy as np
import torch

from src.data_loader import (ROOT, get_input_dim, load_config,
                             load_global_test, load_hospital_data)
from src.metrics import evaluate, evaluate_per_site, fairness_gaps
from src.model import build_model, train


def load_combined_train(num_hospitals: int):
    """Concatenate every hospital's TRAIN split (validation sets stay held out)."""
    parts = [load_hospital_data(h)[:2] for h in range(num_hospitals)]
    return np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])


def run_centralized(cfg=None, save=True) -> dict:
    cfg = cfg or load_config()
    t, H = cfg["training"], cfg["hospitals"]["num_hospitals"]
    torch.manual_seed(cfg["seed"])                     # same seed -> same initial weights
    np.random.seed(cfg["seed"])

    X, y = load_combined_train(H)
    model = build_model(get_input_dim(), cfg)

    start = time.perf_counter()
    train(model, X, y, epochs=cfg["centralized"]["epochs"], lr=t["learning_rate"],
          batch_size=t["batch_size"], cfg=cfg, seed=cfg["seed"])
    train_seconds = time.perf_counter() - start        # training only, no data loading

    Xte, yte = load_global_test()
    per_site = evaluate_per_site(model, range(H), with_ci=True)
    result = {
        "model": cfg["model"],
        "epochs": cfg["centralized"]["epochs"],
        "n_train": int(len(y)),
        "train_seconds": round(train_seconds, 2),
        "global_test": evaluate(model, Xte, yte),
        "per_site_val": per_site,
        "fairness_gaps": fairness_gaps(per_site),
        "loss_history": model.loss_history,
    }

    if save:
        res = ROOT / cfg["paths"]["results_dir"]
        ckpt = ROOT / cfg["paths"]["checkpoints_dir"]
        res.mkdir(parents=True, exist_ok=True)
        ckpt.mkdir(parents=True, exist_ok=True)
        (res / "centralized.json").write_text(json.dumps(result, indent=2))
        torch.save(model.state_dict(), ckpt / "centralized.pt")
    return result


if __name__ == "__main__":
    r = run_centralized()
    print(f"train time: {r['train_seconds']} s on {r['n_train']} rows")
    print("global test:", {k: round(v, 4) for k, v in r["global_test"].items()})
    print("fairness gaps:", {k: round(v, 4) for k, v in r["fairness_gaps"].items()})
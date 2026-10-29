import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from src.data_loader import ROOT, _processed_dir, load_config

BANDS = ["<50", "50-69", "70+"]
MIN_FRAC = 0.05      # every hospital holds at least 5% of the pool
MAX_FRAC = 0.45      # ...and at most 45%
MIN_RATE = 0.04      # positive rate per hospital must lie in [4%, 25%]
MAX_RATE = 0.25
MIN_POS = 200

def age_band(bracket: str) -> str:
    """'[70-80)' -> '70+'. Feature-skew groups."""
    lo = int(bracket.strip("[)").split("-")[0])
    return "<50" if lo < 50 else ("50-69" if lo < 70 else "70+")


def _assign_patients(pat, H, alpha, quantity_skew, rng):
    """Give each patient one hospital. Lower alpha = stronger skew."""
    q = rng.dirichlet(np.full(H, 2.0)) if quantity_skew else np.full(H, 1.0 / H)
    hosp = np.empty(len(pat), dtype=int)
    for _, idx in pat.groupby(["band", "label"]).indices.items():
        p = rng.dirichlet(np.maximum(alpha * H * q, 1e-3))
        hosp[idx] = rng.choice(H, size=len(idx), p=p)
    return hosp


def build_hospitals(cfg=None) -> pd.DataFrame:
    """pool.npz -> hospital_0..H-1.npz (+ results/hospital_partition.csv). Returns summary table."""
    cfg = cfg or load_config()
    H = cfg["hospitals"]["num_hospitals"]
    sk = cfg["hospitals"]["skew"]
    seed = cfg["seed"]
    out = _processed_dir(cfg)

    pool = np.load(out / "pool.npz")
    X, y, pid, age = pool["X"], pool["y"], pool["patient_nbr"], pool["age_group"]
    df = pd.DataFrame({"pid": pid, "y": y, "band": [age_band(a) for a in age]})
    pat = df.groupby("pid").agg(band=("band", "first"), label=("y", "max")).reset_index()

    for attempt in range(200):
        rng = np.random.default_rng(seed + attempt)
        pat["hosp"] = _assign_patients(pat, H, sk["dirichlet_alpha"], sk["quantity_skew"], rng)
        h_of_row = df["pid"].map(pat.set_index("pid")["hosp"]).to_numpy()
        sizes = np.bincount(h_of_row, minlength=H)
        pos = np.bincount(h_of_row, weights=y, minlength=H)
        rate = pos / sizes
        if (sizes.min() >= MIN_FRAC * len(y) and sizes.max() <= MAX_FRAC * len(y)
                and pos.min() >= MIN_POS and rate.min() >= MIN_RATE and rate.max() <= MAX_RATE):
            break
    else:
        raise RuntimeError("No valid partition found; raise dirichlet_alpha in config.yaml")

    rows = []
    for h in range(H):
        idx = np.where(h_of_row == h)[0]
        gss = GroupShuffleSplit(n_splits=1, test_size=cfg["data"]["val_size"], random_state=seed)
        tr, va = next(gss.split(np.zeros(len(idx)), groups=pid[idx]))
        tr, va = idx[tr], idx[va]
        np.savez_compressed(out / f"hospital_{h}.npz",
                            X_train=X[tr], y_train=y[tr], X_val=X[va], y_val=y[va])
        sub = df.iloc[idx]
        rows.append({"hospital": h, "n_total": len(idx), "n_train": len(tr), "n_val": len(va),
                     "positive_rate": round(float(y[idx].mean()), 4),
                     **{f"pct_age_{b}": round(float((sub["band"] == b).mean()), 3) for b in BANDS}})

    summary = pd.DataFrame(rows)
    res = ROOT / cfg["paths"]["results_dir"]
    res.mkdir(parents=True, exist_ok=True)
    summary.to_csv(res / "hospital_partition.csv", index=False)
    return summary


if __name__ == "__main__":
    print(build_hospitals().to_string(index=False))
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
AGE_MAP = {f"[{a}-{a+10})": a + 5 for a in range(0, 100, 10)}
EXCLUDE_DISCHARGE = [11, 13, 14, 19, 20, 21]   # expired / hospice
CATEGORY_CODES = ["admission_type_id", "discharge_disposition_id", "admission_source_id"]


# ---------------------------------------------------------------- config
def load_config(path=None) -> dict:
    path = Path(path) if path else ROOT / "configs" / "config.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def _processed_dir(cfg=None) -> Path:
    cfg = cfg or load_config()
    p = ROOT / cfg["paths"]["processed_dir"]
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------- cleaning
def icd9_group(code) -> str:
    """Collapse an ICD-9 diagnosis code into a broad category."""
    if pd.isna(code):
        return "Unknown"
    code = str(code)
    if code.startswith(("V", "E")):
        return "Other"
    if code.startswith("250"):
        return "Diabetes"
    i = int(float(code))
    if 390 <= i < 460 or i == 785: return "Circulatory"
    if 460 <= i < 520 or i == 786: return "Respiratory"
    if 520 <= i < 580 or i == 787: return "Digestive"
    if 580 <= i < 630 or i == 788: return "Genitourinary"
    if 710 <= i < 740:             return "Musculoskeletal"
    if 800 <= i < 1000:            return "Injury"
    if 140 <= i < 240:             return "Neoplasms"
    return "Other"


def load_and_clean(cfg=None):
    """Raw CSV -> (X, y, meta). X: cleaned features (still unencoded).
    y: 0/1 target. meta: patient_nbr and age_group, used for splitting/partitioning."""
    cfg = cfg or load_config()
    d = cfg["data"]
    df = pd.read_csv(
        ROOT / cfg["paths"]["raw_data"],
        na_values=[d["missing_marker"]],
        keep_default_na=False,   # keep "None" as a real category
        low_memory=False,
    )
    df = df[~df["discharge_disposition_id"].isin(EXCLUDE_DISCHARGE)]
    df = df[df["gender"] != "Unknown/Invalid"]

    y = (df[d["target_source"]] == d["positive_label"]).astype(int)
    meta = pd.DataFrame({"patient_nbr": df["patient_nbr"], "age_group": df["age"]})

    df = df.drop(columns=d["id_columns"] + ["weight", "payer_code", d["target_source"]])
    df["age"] = df["age"].map(AGE_MAP)
    for c in ["diag_1", "diag_2", "diag_3"]:
        df[c] = df[c].map(icd9_group)
    df["race"] = df["race"].fillna("Unknown")
    df["medical_specialty"] = df["medical_specialty"].fillna("Unknown")
    top = df["medical_specialty"].value_counts().nlargest(10).index
    df["medical_specialty"] = df["medical_specialty"].where(
        df["medical_specialty"].isin(top), "Other")
    for c in CATEGORY_CODES:
        df[c] = df[c].astype(str)
    df = df.loc[:, df.nunique() > 1]

    return df.reset_index(drop=True), y.reset_index(drop=True), meta.reset_index(drop=True)


# ---------------------------------------------------------------- split + encode + save
def prepare_pool_and_test(cfg=None) -> dict:
    """Clean -> patient-grouped split -> encode/scale (fit on pool only) -> save.
    Writes pool.npz, global_test.npz, meta.json, preprocessor.joblib."""
    cfg = cfg or load_config()
    X, y, meta = load_and_clean(cfg)

    gss = GroupShuffleSplit(n_splits=1, test_size=cfg["data"]["test_size"],
                            random_state=cfg["seed"])
    pool_idx, test_idx = next(gss.split(X, y, groups=meta["patient_nbr"]))

    num_cols = X.select_dtypes("number").columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    pre = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
    ])
    X_pool = pre.fit_transform(X.iloc[pool_idx]).astype("float32")   # fit on pool only
    X_test = pre.transform(X.iloc[test_idx]).astype("float32")

    out = _processed_dir(cfg)
    np.savez_compressed(
        out / "pool.npz", X=X_pool, y=y.iloc[pool_idx].to_numpy(),
        patient_nbr=meta["patient_nbr"].iloc[pool_idx].to_numpy(),
        age_group=meta["age_group"].iloc[pool_idx].to_numpy().astype(str))
    np.savez_compressed(
        out / "global_test.npz", X=X_test, y=y.iloc[test_idx].to_numpy(),
        patient_nbr=meta["patient_nbr"].iloc[test_idx].to_numpy())
    joblib.dump(pre, out / "preprocessor.joblib")

    info = {"input_dim": int(X_pool.shape[1]),
            "feature_names": pre.get_feature_names_out().tolist(),
            "num_cols": num_cols, "cat_cols": cat_cols,
            "n_pool": int(len(pool_idx)), "n_test": int(len(test_idx))}
    (out / "meta.json").write_text(json.dumps(info, indent=2))
    return info


# ---------------------------------------------------------------- interface for teammate
def load_hospital_data(hospital_id: int):
    """Returns (X_train, y_train, X_val, y_val) as numpy arrays.
    Implemented after partition.py exists."""
    raise NotImplementedError


def load_global_test():
    """Returns (X_test, y_test) as numpy arrays."""
    d = np.load(_processed_dir() / "global_test.npz")
    return d["X"], d["y"]


def get_input_dim() -> int:
    """Number of features after encoding."""
    return json.loads((_processed_dir() / "meta.json").read_text())["input_dim"]


if __name__ == "__main__":
    print(prepare_pool_and_test())
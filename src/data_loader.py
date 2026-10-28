from pathlib import Path
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
AGE_MAP = {f"[{a}-{a+10})": a + 5 for a in range(0, 100, 10)}
EXCLUDE_DISCHARGE = [11, 13, 14, 19, 20, 21]   # expired / hospice
CATEGORY_CODES = ["admission_type_id", "discharge_disposition_id", "admission_source_id"]


def load_config(path=None) -> dict:
    path = Path(path) if path else ROOT / "configs" / "config.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


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
    keep_default_na=False,
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
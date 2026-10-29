import numpy as np
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)

from src.data_loader import load_config, load_hospital_data
from src.model import predict_proba


def compute_metrics(y_true, y_prob, threshold=0.5) -> dict:
    """accuracy, precision, recall, f1, roc_auc from true labels and probabilities."""
    y_true = np.asarray(y_true)
    y_pred = (y_prob >= threshold).astype(int)
    both = len(np.unique(y_true)) == 2          # ROC-AUC needs both classes present
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if both else float("nan"),
        "n": int(len(y_true)),
        "n_pos": int(y_true.sum()),
    }


def evaluate(model, X, y, threshold=None) -> dict:
    """Score a model on (X, y). Threshold defaults to training.threshold in config."""
    if threshold is None:
        threshold = load_config()["training"]["threshold"]
    return compute_metrics(y, predict_proba(model, X), threshold)


def bootstrap_auc_ci(y, y_prob, n_boot=300, seed=0) -> tuple:
    """95% bootstrap interval for ROC-AUC. Small hospitals have noisy estimates."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    aucs = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) == 2:
            aucs.append(roc_auc_score(y[i], y_prob[i]))
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def evaluate_per_site(model, hospital_ids, threshold=None, with_ci=False) -> dict:
    """Metrics on each hospital's own validation set -> {hospital_id: metrics dict}.
    Validation sets are never used for training, so this is a fair per-site score."""
    if threshold is None:
        threshold = load_config()["training"]["threshold"]
    out = {}
    for h in hospital_ids:
        _, _, Xva, yva = load_hospital_data(h)
        p = predict_proba(model, Xva)
        out[h] = compute_metrics(yva, p, threshold)
        if with_ci:
            out[h]["roc_auc_ci"] = bootstrap_auc_ci(yva, p)
    return out


def fairness_gaps(per_site: dict) -> dict:
    """Max minus min of each metric across sites (smaller = fairer)."""
    gaps = {}
    for m in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        v = [s[m] for s in per_site.values() if not np.isnan(s[m])]
        gaps[m] = float(max(v) - min(v))
    return gaps
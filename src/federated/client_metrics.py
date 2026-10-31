"""NumPy-only versions of src.metrics.compute_metrics and bootstrap_auc_ci, for the ClientApp.

Same definitions and outputs as src/metrics.py (accuracy, precision, recall, f1 with
zero_division=0, ROC-AUC, and the same bootstrap draws), but without importing scikit-learn,
which loads SciPy, joblib and pandas and costs ~100 MB in every client process. Flower starts a
new client process per message, so this matters when five hospitals share one small machine.
docker/check_client_metrics.py verifies that both versions give identical results.
"""
import numpy as np


def roc_auc(y_true, y_prob) -> float:
    """ROC-AUC as the Mann-Whitney statistic with average ranks for ties (equals sklearn's)."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    n1 = int(y_true.sum())
    n0 = len(y_true) - n1
    order = np.argsort(y_prob, kind="mergesort")
    s = y_prob[order]
    ranks = np.empty(len(s), dtype=np.float64)
    # average rank (1-based) over each block of tied scores
    starts = np.r_[0, np.flatnonzero(np.diff(s)) + 1]
    ends = np.r_[starts[1:], len(s)]
    avg = (starts + ends + 1) / 2.0
    ranks[order] = np.repeat(avg, ends - starts)
    return float((ranks[y_true == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def compute_metrics(y_true, y_prob, threshold=0.5) -> dict:
    """Same keys and values as src.metrics.compute_metrics."""
    y_true = np.asarray(y_true)
    y_pred = (y_prob >= threshold).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    both = len(np.unique(y_true)) == 2
    return {
        "accuracy": float((y_pred == y_true).mean()),
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "f1": 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0,
        "roc_auc": roc_auc(y_true, y_prob) if both else float("nan"),
        "n": int(len(y_true)),
        "n_pos": int(y_true.sum()),
    }


def bootstrap_auc_ci(y, y_prob, n_boot=300, seed=0) -> tuple:
    """Same draws and percentiles as src.metrics.bootstrap_auc_ci."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    aucs = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) == 2:
            aucs.append(roc_auc(y[i], y_prob[i]))
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))

def evaluate(model, X, y, threshold=0.5) -> dict:
    """accuracy, precision, recall, f1, roc_auc"""
    raise NotImplementedError
def evaluate_per_site(model, hospital_ids) -> dict:
    """metrics per hospital, used for the fairness analysis"""
    raise NotImplementedError
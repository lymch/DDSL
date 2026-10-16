"""Image-level detection metrics only."""
import numpy as np
from sklearn.metrics import roc_auc_score


def detection_aurocs(scores, labels, paths, logical=True):
    scores = np.asarray(scores).reshape(-1)
    labels = np.asarray(labels).reshape(-1)
    if not (len(scores) == len(labels) == len(paths)):
        raise ValueError("Scores, labels and paths must have equal lengths.")
    if not np.all(np.isfinite(scores)):
        raise ValueError("Anomaly scores must be finite.")
    if not logical:
        return {"auroc_struct": float(roc_auc_score(labels, scores))}
    types = np.asarray([str(path).replace("\\", "/").split("/")[-2] for path in paths])
    result = {}
    for name, defect in (("auroc_logic", "logical_anomalies"),
                         ("auroc_struct", "structural_anomalies")):
        selected = (types == "good") | (types == defect)
        if np.unique(labels[selected]).size != 2:
            raise ValueError("AUROC requires normal and anomalous samples for " + defect)
        result[name] = float(roc_auc_score(labels[selected], scores[selected]))
    return result

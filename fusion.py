"""Normal-validation score calibration and max fusion."""
import numpy as np


def image_key(path):
    return tuple(str(path).replace("\\", "/").split("/")[-2:])


def align_scores(scores, paths, target_paths):
    keys = [image_key(path) for path in paths]
    targets = [image_key(path) for path in target_paths]
    scores = np.asarray(scores).reshape(-1)
    if len(keys) != len(scores) or len(set(keys)) != len(keys):
        raise ValueError("Duplicate image keys or inconsistent prediction lengths.")
    if len(set(targets)) != len(targets) or set(keys) != set(targets):
        raise ValueError("Appearance and composition image sets do not match.")
    lookup = dict(zip(keys, scores))
    return np.asarray([lookup[key] for key in targets])


def fit_normalization(scores, labels):
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels).reshape(-1)
    if not len(scores) or len(scores) != len(labels):
        raise ValueError("Calibration requires nonempty matching scores and labels.")
    if np.any(labels != 0) or not np.all(np.isfinite(scores)):
        raise ValueError("Calibration requires finite scores from normal-only validation data.")
    return float(scores.mean()), float(scores.std())


def normalize_branch_scores(scores, mean, std, eps=1e-12):
    return (np.asarray(scores, dtype=np.float64) - mean) / max(std, eps)


def fuse_predictions(appearance, composition, appearance_stats, composition_stats):
    app_scores, app_labels, app_paths = appearance
    comp_scores, comp_labels, comp_paths = composition
    app_scores = align_scores(app_scores, app_paths, comp_paths)
    app_labels = align_scores(app_labels, app_paths, comp_paths)
    if not np.array_equal(app_labels, comp_labels):
        raise ValueError("Branch labels disagree for matched images.")
    scores = np.maximum(
        normalize_branch_scores(app_scores, *appearance_stats),
        normalize_branch_scores(comp_scores, *composition_stats),
    )
    return scores, comp_labels, comp_paths

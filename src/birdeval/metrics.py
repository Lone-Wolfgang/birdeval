"""Multi-label metrics, following BirdSet's reporting (cmAP, AUROC, top-1).

All metrics are threshold-free. Classes with no positives in the eval set are
excluded from class-wise averages (AP and AUROC are undefined for them).

`restrict` lets you score on a subset of classes, e.g. the classes every model
in a comparison can actually predict, so a model isn't penalised (or flattered)
by vocabulary differences.

Ground truth is three-state (1 present, 0 absent, -1 not reviewed). Each class
is scored only on the rows where it is known, so an unreviewed call the model
finds is neither a hit nor a false positive. With no -1 cells every metric is
identical to the binary computation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def _valid_classes(y: np.ndarray, restrict: np.ndarray | None) -> np.ndarray:
    has_pos = (y == 1).any(axis=0)
    has_neg = (y == 0).any(axis=0)
    mask = has_pos & has_neg
    if restrict is not None:
        mask &= restrict
    return np.flatnonzero(mask)


def compute(
    y: np.ndarray, s: np.ndarray, classes: list[str], restrict: np.ndarray | None = None
) -> tuple[dict, pd.DataFrame]:
    cols = _valid_classes(y, restrict)
    known = [y[:, c] != -1 for c in cols]
    per_class = pd.DataFrame(
        {
            "class": [classes[c] for c in cols],
            "n_pos": (y[:, cols] == 1).sum(axis=0),
            "n_scored": [int(k.sum()) for k in known],
            "ap": [average_precision_score(y[k, c], s[k, c]) for c, k in zip(cols, known)],
            "auroc": [roc_auc_score(y[k, c], s[k, c]) for c, k in zip(cols, known)],
        }
    )
    # top-1: among segments with >=1 label, is the top-scoring class a true label?
    # Rows whose top class is unreviewed say nothing either way and are skipped.
    use = np.ones(len(classes), bool) if restrict is None else restrict
    rows = np.flatnonzero((y[:, use] == 1).any(axis=1))
    top = np.argmax(np.where(use, s, -np.inf)[rows], axis=1)
    hit = y[rows, top]
    hit = hit[hit != -1]
    top1 = float((hit == 1).mean()) if len(hit) else float("nan")
    summary = {
        "cmAP": float(per_class["ap"].mean()) if len(per_class) else float("nan"),
        "AUROC": float(per_class["auroc"].mean()) if len(per_class) else float("nan"),
        "top1": top1,
        "n_segments": int(len(y)),
        "n_classes_scored": int(len(cols)),
    }
    return summary, per_class


def bootstrap_cmap(
    y: np.ndarray,
    s: np.ndarray,
    groups: np.ndarray,
    restrict: np.ndarray | None = None,
    n: int = 200,
    seed: int = 0,
) -> tuple[float, float]:
    """95% CI for cmAP, resampling whole files (segments within a file are correlated)."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    by_group = {g: np.flatnonzero(groups == g) for g in uniq}
    vals = []
    for _ in range(n):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([by_group[g] for g in pick])
        yb, sb = y[idx], s[idx]
        cols = _valid_classes(yb, restrict)
        if len(cols):
            aps = []
            for c in cols:
                k = yb[:, c] != -1
                aps.append(average_precision_score(yb[k, c], sb[k, c]))
            vals.append(np.mean(aps))
    if not vals:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))

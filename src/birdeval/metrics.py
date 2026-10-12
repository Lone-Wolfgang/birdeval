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

import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy.stats import rankdata


def default_workers() -> int:
    return min(8, os.cpu_count() or 1)


_STATE = None  # the object a pool worker calls into, sent once per worker


def _set_state(obj) -> None:
    global _STATE
    _STATE = obj


def _call(args):
    method, chunk = args
    return [getattr(_STATE, method)(x) for x in chunk]


def parallel_map(obj, method: str, items: list, workers: int = 1) -> list:
    """obj.method over items, in order, on up to `workers` processes.

    The bootstraps are bound by memory bandwidth, so threads don't help;
    each spawned worker gets `obj` (the presorted arrays) once.
    """
    workers = min(workers, len(items))
    if workers <= 1:
        return [getattr(obj, method)(x) for x in items]
    chunks = [items[i::workers] for i in range(workers)]
    with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn"),
                             initializer=_set_state, initargs=(obj,)) as pool:
        done = list(pool.map(_call, [(method, c) for c in chunks]))
    out = [None] * len(items)
    for i, res in enumerate(done):
        out[i::workers] = res
    return out


def auroc(t: np.ndarray, s: np.ndarray) -> float:
    """Mann-Whitney AUROC with ties counted half; equals sklearn's roc_auc_score."""
    pos = t == 1
    n1, n0 = pos.sum(), (~pos).sum()
    r = rankdata(np.asarray(s, np.float64))  # float32 scores would get float32 ranks and a lossy sum
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def ap(t: np.ndarray, s: np.ndarray) -> float:
    """Average precision as in sklearn: precision at each distinct threshold, weighted by recall gained."""
    o = np.argsort(-s, kind="stable")
    ss, tt = s[o], t[o] == 1
    last = np.r_[ss[1:] != ss[:-1], True]
    tp = np.cumsum(tt)[last]
    n = np.flatnonzero(last) + 1
    rec = tp / tp[-1]
    return float(np.sum(np.diff(np.r_[0, rec]) * tp / n))


class SortedColumns:
    """Every class column sorted by score once, so a bootstrap draw is a reweighting.

    A draw that picks a row k times is the same as weighting it k; AP over the
    resample is then sum_i w_i y_i P(end of i's tie group) / sum_i w_i y_i,
    with P = cumsum(w y) / cumsum(w). Unknown cells (-1) get weight 0.
    """

    def __init__(self, y: np.ndarray, s: np.ndarray, cols: np.ndarray):
        self.cols = cols
        sc = s[:, cols]
        order = np.argsort(-sc, axis=0, kind="stable")
        ss = np.take_along_axis(sc, order, axis=0)
        ys = np.take_along_axis(y[:, cols], order, axis=0)
        self.order = order.astype(np.int32)  # int32 throughout: the draws are bandwidth-bound
        self.pos, self.known = (ys == 1).astype(np.int32), (ys != -1).astype(np.int32)
        # index of the last row of each row's tie group, per column
        n = len(ss)
        is_last = np.vstack([ss[1:] != ss[:-1], np.ones((1, len(cols)), bool)])
        idx = np.where(is_last, np.arange(n, dtype=np.int32)[:, None], np.int32(n))
        self.end = np.minimum.accumulate(idx[::-1], axis=0)[::-1]

    def ap(self, row_w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(AP per column, valid: weighted positives and negatives both > 0). row_w: int counts."""
        w = row_w.astype(np.int32)[self.order] * self.known
        wp = w * self.pos
        tp, n = np.cumsum(wp, axis=0, dtype=np.int32), np.cumsum(w, axis=0, dtype=np.int32)
        n_pos = tp[-1]
        valid = (n_pos > 0) & (n[-1] > n_pos)
        prec = np.divide(tp, n, out=np.zeros(tp.shape), where=n > 0)
        p_end = np.take_along_axis(prec, self.end, axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            return (wp * p_end).sum(axis=0) / n_pos, valid


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
            "ap": [ap(y[k, c], s[k, c]) for c, k in zip(cols, known)],
            "auroc": [auroc(y[k, c], s[k, c]) for c, k in zip(cols, known)],
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
    workers: int = 1,
) -> tuple[float, float]:
    """95% CI for cmAP, resampling whole files (segments within a file are correlated).

    Each draw is a weighting of the rows (how often each row's group was
    drawn), so the columns are sorted once rather than per draw.
    """
    rng = np.random.default_rng(seed)
    uniq, g_idx = np.unique(groups, return_inverse=True)
    draws = [np.bincount(rng.choice(len(uniq), size=len(uniq), replace=True), minlength=len(uniq))
             for _ in range(n)]
    cols = np.arange(y.shape[1]) if restrict is None else np.flatnonzero(restrict)
    if not len(cols):
        return float("nan"), float("nan")
    sc = SortedColumns(y, s, cols)

    draws = [counts[g_idx] for counts in draws]
    vals = [v for v in parallel_map(_CmapDraw(sc), "mean_ap", draws, workers) if v is not None]
    if not vals:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


class _CmapDraw:
    def __init__(self, sc: SortedColumns):
        self.sc = sc

    def mean_ap(self, row_w: np.ndarray) -> float | None:
        aps, valid = self.sc.ap(row_w)
        return float(aps[valid].mean()) if valid.any() else None

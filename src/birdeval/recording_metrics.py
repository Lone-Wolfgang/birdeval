"""Recording-level, precision-first metrics on (unit, species) presence claims.

Only pairs with known truth (0 or 1) count. Pairs are restricted to a class
scope exactly as at segment level (all / covered / shared).

- rec_cmAP: per-species AP across units, averaged over species with at least
  one present and one absent unit.
- recall at precision >= target: pooled over all pairs, the lowest threshold
  (= highest recall) whose precision meets the target. If none does, recall
  is 0 and `reached` is False.
- list precision at that operating threshold: of the pairs claimed (score >=
  threshold), the fraction present. Pooled (micro) and the mean over units
  with at least one claim.
- by duration: the same at the *global* operating threshold, per duration
  bin. Re-picking a threshold per bin would hide that longer recordings give
  absent species more chances at one high window.
- bootstrap CIs resample groups (sites if the eval set has a `site` column,
  else files). Recall-at-precision re-picks its threshold in each resample;
  list precision is measured at the fixed global threshold, as deployed.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score


def recall_at_precision(truth: np.ndarray, score: np.ndarray, target: float) -> dict:
    """truth/score: 1-D over known pairs (truth in {0, 1})."""
    n_pos = int((truth == 1).sum())
    out = {"target": target, "threshold": float("nan"), "precision": float("nan"),
           "recall": 0.0, "n_claims": 0, "reached": False}
    if n_pos == 0 or len(truth) == 0:
        return out
    order = np.argsort(-score, kind="stable")
    s, t = score[order], (truth[order] == 1)
    tp = np.cumsum(t)
    # only cut between distinct scores: tied pairs are claimed together
    last = np.r_[s[1:] != s[:-1], True]
    tp, n = tp[last], np.flatnonzero(last) + 1
    prec, rec = tp / n, tp / n_pos
    ok = np.flatnonzero(prec >= target)
    if not len(ok):
        return out
    i = ok[np.argmax(rec[ok])]  # highest recall; on ties, the earliest (highest threshold)
    i = ok[rec[ok] == rec[i]][0]
    return {"target": target, "threshold": float(s[last][i]), "precision": float(prec[i]),
            "recall": float(rec[i]), "n_claims": int(n[i]), "reached": True}


def list_precision(truth: np.ndarray, score: np.ndarray, unit: np.ndarray, threshold: float) -> dict:
    """Precision of the claimed species lists at `threshold`, pooled and per unit."""
    claim = score >= threshold if np.isfinite(threshold) else np.zeros(len(score), bool)
    tp = (truth == 1) & claim
    n_pos = int((truth == 1).sum())
    micro = float(tp.sum() / claim.sum()) if claim.any() else float("nan")
    units = np.unique(unit[claim])
    per_unit = [tp[unit == u].sum() / claim[unit == u].sum() for u in units]
    return {
        "list_precision": micro,
        "list_precision_unit_mean": float(np.mean(per_unit)) if per_unit else float("nan"),
        "recall": float(tp.sum() / n_pos) if n_pos else float("nan"),
        "n_claims": int(claim.sum()),
        "n_units_with_claims": int(len(units)),
    }


def rec_cmap(T: np.ndarray, S: np.ndarray, cols: np.ndarray) -> tuple[float, int]:
    aps = []
    for c in cols:
        k = T[:, c] != -1
        y = T[k, c]
        if (y == 1).any() and (y == 0).any():
            aps.append(average_precision_score(y, S[k, c]))
    return (float(np.mean(aps)) if aps else float("nan")), len(aps)


def _pairs(T: np.ndarray, S: np.ndarray, cols: np.ndarray, rows: np.ndarray | None = None):
    """Flatten known (unit, class) pairs -> (truth, score, unit position in `rows`).

    Positions (not unit ids) keep a unit drawn twice by the bootstrap distinct.
    """
    rows = np.arange(len(T)) if rows is None else rows
    t, s = T[np.ix_(rows, cols)], S[np.ix_(rows, cols)]
    u = np.repeat(np.arange(len(rows)), len(cols)).reshape(t.shape)
    k = t != -1
    return t[k], s[k], u[k]


def compute(
    T: np.ndarray,
    S: np.ndarray,
    duration_s: np.ndarray,
    groups: np.ndarray,
    restrict: np.ndarray | None = None,
    targets: list[float] = (0.9,),
    duration_bins_s: list[float] = (60, 300, 900, 3600),
    n_boot: int = 200,
    seed: int = 0,
) -> dict:
    """T: [U, C] int8 unit truth; S: [U, C] presence scores."""
    cols = np.arange(T.shape[1]) if restrict is None else np.flatnonzero(restrict)
    cmap, n_cls = rec_cmap(T, S, cols)
    t, s, u = _pairs(T, S, cols)
    out = {"cmAP": cmap, "n_classes_scored": n_cls, "n_units": int(len(T)),
           "n_pairs": int(len(t)), "n_present": int((t == 1).sum()), "targets": {}}

    edges = np.r_[0.0, np.asarray(duration_bins_s, float), np.inf]
    bin_of = np.digitize(duration_s, edges[1:-1], right=True)
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    by_group = {g: np.flatnonzero(groups == g) for g in uniq}
    boots = [np.concatenate([by_group[g] for g in rng.choice(uniq, len(uniq))]) for _ in range(n_boot)]

    for target in targets:
        op = recall_at_precision(t, s, target)
        lp = list_precision(t, s, u, op["threshold"])
        res = {**op, "list_precision": lp["list_precision"],
               "list_precision_unit_mean": lp["list_precision_unit_mean"],
               "n_units_with_claims": lp["n_units_with_claims"]}
        if not op["reached"]:
            res["note"] = f"no threshold reaches precision {target}"
        if boots:
            rec_b, lp_b = [], []
            for idx in boots:
                tb, sb, ub = _pairs(T, S, cols, idx)
                rec_b.append(recall_at_precision(tb, sb, target)["recall"])
                lp_b.append(list_precision(tb, sb, ub, op["threshold"])["list_precision"])
            res["recall_ci95"] = _ci(rec_b)
            res["list_precision_ci95"] = _ci(lp_b)
        res["by_duration"] = []
        for b in range(len(edges) - 1):
            rows = np.flatnonzero(bin_of == b)
            if not len(rows):
                continue
            tb, sb, ub = _pairs(T, S, cols, rows)
            lpb = list_precision(tb, sb, ub, op["threshold"])
            res["by_duration"].append({
                "bin_s": [float(edges[b]), float(edges[b + 1])],
                "n_units": int(len(rows)),
                "cmAP": rec_cmap(T[rows], S[rows], cols)[0],
                "recall": lpb["recall"],
                "list_precision": lpb["list_precision"],
                "list_precision_unit_mean": lpb["list_precision_unit_mean"],
            })
        out["targets"][str(target)] = res
    return out


def _ci(vals: list[float]) -> tuple[float, float]:
    v = np.asarray(vals, float)
    v = v[np.isfinite(v)]
    if not len(v):
        return float("nan"), float("nan")
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))

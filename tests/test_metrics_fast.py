"""The weighted bootstraps equal copying and re-scoring each resample; numpy AP/AUROC equal sklearn."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from birdeval import metrics, recording_metrics
from birdeval.recording_metrics import _ci, _pairs, list_precision, recall_at_precision


def _data(seed, n=240, c=7, unknown=0.05):
    rng = np.random.default_rng(seed)
    y = (rng.random((n, c)) < 0.15).astype(np.int8)
    y[rng.random(y.shape) < unknown] = -1
    y[:, 5] = 0  # a class with no positives
    s = (y + rng.normal(0, 0.8, y.shape)).astype(np.float32)
    s[: n // 4] = np.round(s[: n // 4])  # ties
    return y, s, np.repeat(np.arange(n // 8), 8)


def test_numpy_ap_auroc_equal_sklearn():
    y, s, _ = _data(0)
    for c in range(4):
        k = y[:, c] != -1
        assert metrics.ap(y[k, c], s[k, c]) == pytest.approx(average_precision_score(y[k, c], s[k, c]), abs=1e-12)
        assert metrics.auroc(y[k, c], s[k, c]) == pytest.approx(roc_auc_score(y[k, c], s[k, c]), abs=1e-12)


def test_auroc_exact_for_float32_scores():
    # scipy ranks float32 input in float32; summing 1500 such ranks loses ~1e-7
    rng = np.random.default_rng(0)
    y = (rng.random(25000) < 0.06).astype(np.int8)
    s = (y + rng.normal(0, 1, len(y))).astype(np.float32)
    assert metrics.auroc(y, s) == pytest.approx(roc_auc_score(y, s), abs=1e-12)


def _copy_bootstrap_cmap(y, s, groups, restrict, n, seed=0):
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    by = {g: np.flatnonzero(groups == g) for g in uniq}
    vals = []
    for _ in range(n):
        idx = np.concatenate([by[g] for g in rng.choice(uniq, size=len(uniq), replace=True)])
        yb, sb = y[idx], s[idx]
        cols = metrics._valid_classes(yb, restrict)
        if len(cols):
            aps = []
            for c in cols:
                k = yb[:, c] != -1
                aps.append(average_precision_score(yb[k, c], sb[k, c]))
            vals.append(np.mean(aps))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("restrict", [None, np.array([1, 1, 0, 1, 1, 1, 1], bool)])
def test_weighted_bootstrap_cmap_equals_copies(seed, restrict):
    y, s, groups = _data(seed)
    want = _copy_bootstrap_cmap(y, s, groups, restrict, n=40)
    for workers in (1, 4):
        got = metrics.bootstrap_cmap(y, s, groups, restrict, n=40, workers=workers)
        assert got == pytest.approx(want, abs=1e-12)


def _copy_recording_ci(T, S, groups, cols, target, n_boot, seed=0):
    t, s, u = _pairs(T, S, cols)
    thr = recall_at_precision(t, s, target)["threshold"]
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    by = {g: np.flatnonzero(groups == g) for g in uniq}
    rec, lp = [], []
    for _ in range(n_boot):
        idx = np.concatenate([by[g] for g in rng.choice(uniq, len(uniq))])
        tb, sb, ub = _pairs(T, S, cols, idx)
        rec.append(recall_at_precision(tb, sb, target)["recall"])
        lp.append(list_precision(tb, sb, ub, thr)["list_precision"])
    return _ci(rec), _ci(lp)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_weighted_recording_bootstrap_equals_copies(seed):
    T, S, _ = _data(seed, n=160)
    groups = np.repeat(np.arange(40), 4)
    targets = [0.5, 0.8, 0.9, 0.99]  # 0.99 is usually never reached
    cols = np.arange(T.shape[1])
    for workers in (1, 3):
        r = recording_metrics.compute(T, S, np.full(len(T), 60.0), groups, targets=targets, n_boot=40,
                                      workers=workers)
        for target in targets:
            rec, lp = _copy_recording_ci(T, S, groups, cols, target, 40)
            got = r["targets"][str(target)]
            np.testing.assert_allclose(got["recall_ci95"], rec, atol=1e-12)
            np.testing.assert_allclose(got["list_precision_ci95"], lp, atol=1e-12, equal_nan=True)

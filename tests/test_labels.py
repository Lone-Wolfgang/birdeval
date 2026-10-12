"""Phase 2: three-state labels (1 present, 0 absent, -1 not reviewed)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from birdeval import metrics
from birdeval.data import EvalSet
from birdeval.evalsets.strong_labels import load_strong_labels


def _old_compute(y, s, restrict=None):
    """metrics.compute / bootstrap_cmap as they were before three-state labels."""
    has = (y.sum(0) > 0) & ((1 - y).sum(0) > 0)
    cols = np.flatnonzero(has if restrict is None else has & restrict)
    use = np.ones(y.shape[1], bool) if restrict is None else restrict
    rows = y[:, use].sum(1) > 0
    top = np.argmax(np.where(use, s, -np.inf)[rows], axis=1)
    return {
        "cmAP": np.mean([average_precision_score(y[:, c], s[:, c]) for c in cols]),
        "AUROC": np.mean([roc_auc_score(y[:, c], s[:, c]) for c in cols]),
        "top1": y[rows][np.arange(rows.sum()), top].mean(),
        "n_classes_scored": len(cols),
    }


def _old_bootstrap(y, s, groups, restrict, n, seed=0):
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    by = {g: np.flatnonzero(groups == g) for g in uniq}
    vals = []
    for _ in range(n):
        idx = np.concatenate([by[g] for g in rng.choice(uniq, size=len(uniq), replace=True)])
        has = (y[idx].sum(0) > 0) & ((1 - y[idx]).sum(0) > 0)
        cols = np.flatnonzero(has if restrict is None else has & restrict)
        if len(cols):
            vals.append(np.mean([average_precision_score(y[idx, c], s[idx, c]) for c in cols]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


@pytest.mark.parametrize("restrict", [None, np.array([True, True, False, True, True, True])])
def test_defaults_match_binary_metrics_exactly(restrict):
    rng = np.random.default_rng(0)
    y = (rng.random((300, 6)) < 0.15).astype(np.int8)
    s = (y + rng.normal(0, 0.8, y.shape)).astype(np.float32)
    s[:20] = np.round(s[:20])  # some ties too
    groups = np.repeat(np.arange(30), 10)
    summ, _ = metrics.compute(y, s, list("abcdef"), restrict=restrict)
    old = _old_compute(y.astype(np.uint8), s, restrict)
    for k, v in old.items():
        assert summ[k] == v, k  # exact, not approx
    # the bootstrap reweights presorted columns: equal up to float rounding
    assert metrics.bootstrap_cmap(y, s, groups, restrict, n=30) == pytest.approx(
        _old_bootstrap(y.astype(np.uint8), s, groups, restrict, n=30), abs=1e-12
    )


def _ev(tmp_path, toy_rows, **opts):
    import soundfile as sf

    for f in {r[0] for r in toy_rows} | {"empty.wav"}:
        sf.write(tmp_path / f, np.zeros(16_000 * 20, np.float32), 16_000)  # 20 s
    ann = tmp_path / "ann.csv"
    pd.DataFrame(toy_rows, columns=["file", "start", "end", "label"]).to_csv(ann, index=False)
    return load_strong_labels("t", _Tax(), str(ann), audio_root=str(tmp_path), **opts)


class _Tax:
    def map_many(self, labels, kind):
        return [l.lower() for l in labels], []


ROWS = [("a.wav", 1, 2, "Turdus merula"), ("a.wav", 11, 12, "Parus major"), ("b.wav", 6, 7, "Turdus merula")]


def test_default_loader_has_no_unknowns(tmp_path):
    ev = _ev(tmp_path, ROWS)
    assert "unreviewed" not in ev.segments
    assert set(np.unique(ev.target_matrix())) <= {0, 1}
    np.testing.assert_array_equal(ev.target_matrix_binary(), ev.target_matrix())


def test_unreviewed_species_drops_out_of_scoring(tmp_path):
    full = _ev(tmp_path, ROWS)
    part = _ev(tmp_path, ROWS, reviewed_species=["Turdus merula"])
    y = part.target_matrix()
    pm = part.classes.index("parus major")
    # a positive annotation still counts; everything else for Parus major is unknown
    assert (y[:, pm] == 1).sum() == 1 and (y[:, pm] == 0).sum() == 0
    rng = np.random.default_rng(0)
    s = rng.random(y.shape)
    assert metrics.compute(full.target_matrix(), s, full.classes)[0]["n_classes_scored"] == 2
    assert metrics.compute(y, s, part.classes)[0]["n_classes_scored"] == 1


def test_unreviewed_time_drops_rows(tmp_path):
    ranges = tmp_path / "rev.csv"
    # a.wav reviewed 0-10 s only; b.wav fully; empty.wav fully (reviewed, no birds)
    pd.DataFrame(
        [("a.wav", 0, 10), ("b.wav", 0, 20), ("empty.wav", 0, 7), ("empty.wav", 6, 20)],
        columns=["file", "start", "end"],
    ).to_csv(ranges, index=False)
    ev = _ev(tmp_path, ROWS, reviewed_ranges=str(ranges))
    seg = ev.segments
    assert len(ev.files) == 3  # empty.wav joins as a negative file
    y = ev.target_matrix()
    a_late = ((seg.file.str.endswith("a.wav")) & (seg.start >= 10)).to_numpy()
    pm, tm = ev.classes.index("parus major"), ev.classes.index("turdus merula")
    assert y[a_late, pm].tolist() == [1, -1]  # the call at 11 s still counts
    assert (y[a_late, tm] == -1).all()
    assert (y[~a_late] != -1).all()  # overlapping ranges on empty.wav merge to full coverage
    _, pc = metrics.compute(y, np.random.default_rng(0).random(y.shape), ev.classes)
    assert pc.set_index("class").loc["turdus merula", "n_scored"] == len(y) - a_late.sum()


def test_unreviewed_call_found_by_model_is_not_a_false_positive():
    # 1 true positive, 1 unlabelled real call, 8 true negatives
    s = np.array([0.9, 0.95, 0.1, 0.2, 0.3, 0.1, 0.2, 0.15, 0.05, 0.25])[:, None]
    as_absent = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0], np.int8)[:, None]
    as_unknown = as_absent.copy()
    as_unknown[1] = -1
    ap_absent = metrics.compute(as_absent, s, ["x"])[0]["cmAP"]
    ap_unknown = metrics.compute(as_unknown, s, ["x"])[0]["cmAP"]
    assert ap_absent == pytest.approx(0.5) and ap_unknown == pytest.approx(1.0)


def test_top1_skips_rows_whose_top_class_is_unknown():
    y = np.array([[1, -1], [1, 0]], np.int8)
    s = np.array([[0.1, 0.9], [0.1, 0.9]])
    assert metrics.compute(y, s, ["a", "b"])[0]["top1"] == 0.0  # row 0 skipped, row 1 wrong


def test_evalset_rejects_unknown_unreviewed_class():
    seg = pd.DataFrame({"file": ["f"], "start": [0.0], "end": [5.0], "labels": [[]], "unreviewed": [["zz"]]})
    with pytest.raises(ValueError, match="unreviewed"):
        EvalSet("t", seg, ["a"])

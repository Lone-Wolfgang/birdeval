"""Phase 4: evidence precision, detections.csv and fp_review.csv."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from birdeval import recording_metrics as rm
from birdeval.aggregate import aggregate, evidence_truth, make_units, unit_truth
from birdeval.data import EvalSet, WindowScores
from birdeval.models.fake import FakeModel
from birdeval.runner import evaluate

from conftest import SPECIES

F = "/x/f.wav"


def _planted(unreviewed_a_at_10: bool = False):
    # 20 s file, 5 s segments: A calls at 0-5, B at 10-15
    seg = pd.DataFrame({"file": [F] * 4, "start": [0, 5, 10, 15.0], "end": [5, 10, 15, 20.0],
                        "labels": [["a"], [], ["b"], []]})
    if unreviewed_a_at_10:
        seg["unreviewed"] = [[], [], ["a"], []]
    ev = EvalSet("p", seg, ["a", "b"])
    # A's best window is 10-15, where only B is labelled
    sc = np.array([[2.0, -5], [-5, -5], [6, 4], [-5, -5]], np.float32)
    ws = WindowScores(F, np.array([0, 5, 10, 15.0]), np.array([5, 10, 15, 20.0]), sc, ["a", "b"])
    units, seg_unit = make_units(ev)
    y = ev.target_matrix()
    us = aggregate(ev, {F: ws}, units)
    return y, us, unit_truth(y, seg_unit, len(units)), evidence_truth(ev, y, us)


def test_right_species_wrong_clip():
    _, us, T, E = _planted()
    np.testing.assert_array_equal(T, [[1, 1]])  # both species are in the recording
    np.testing.assert_array_equal(E, [[0, 1]])  # but A's cited clip is B's call
    r = rm.evidence_precision(T, us.max_score, E, np.arange(2), threshold=3.0)
    assert r["evidence_precision"] == 0.5 and r["conditional_evidence_precision"] == 0.5


def test_unreviewed_evidence_is_excluded():
    _, us, T, E = _planted(unreviewed_a_at_10=True)
    assert E[0, 0] == -1
    r = rm.evidence_precision(T, us.max_score, E, np.arange(2), threshold=3.0)
    assert r["n_evidence_known"] == 1 and r["n_evidence_unknown"] == 1
    assert r["evidence_precision"] == 1.0


def test_wrong_species_vs_wrong_clip():
    # conditional precision ignores claims that are wrong at unit level
    T = np.array([[1, 0]], np.int8)
    E = np.array([[1, 0]], np.int8)
    r = rm.evidence_precision(T, np.array([[5.0, 5.0]]), E, np.arange(2), threshold=1.0)
    assert r["evidence_precision"] == 0.5 and r["conditional_evidence_precision"] == 1.0


@pytest.fixture
def run(toy, tmp_path):
    _, tax, ev = toy
    m = FakeModel("noisy", tax, SPECIES, noise=0.7, seed=4)
    m.oracle = ev
    cfg = {"output_dir": str(tmp_path / "res"), "cache_dir": str(tmp_path / "cache"), "bootstrap": 0,
           "recording_level": {"precision_targets": [0.5], "operating_target": 0.5, "fp_review_n": 5}}
    evaluate(cfg, [], [ev], run_name="e", models={"noisy": m})
    return tmp_path / "res" / "e" / "toy" / "noisy"


def test_fp_review_is_sorted_wrong_claims_only(run):
    fp = pd.read_csv(run / "fp_review.csv")
    pres = pd.read_csv(run / "presence.csv")
    assert 0 < len(fp) <= 5
    assert fp["presence_conf"].is_monotonic_decreasing
    truth = fp.merge(pres, on=["unit", "species"])["truth"]
    assert (truth == 0).all()
    # and they really are the top-scoring absent pairs
    top = pres[pres.truth == 0]["presence_conf"].nlargest(len(fp)).to_numpy()
    np.testing.assert_allclose(fp["presence_conf"], top)


def test_detections_table(run):
    det = pd.read_csv(run / "detections.csv")
    assert list(det.columns) == ["file", "start", "end", "species", "confidence", "unit",
                                 "presence_correct", "evidence_correct"]
    assert len(det) and set(det.presence_correct) <= {0, 1, -1}
    assert (det.end > det.start).all()
    rec = json.loads((run / "recording_level.json").read_text())["all"]["targets"]["0.5"]
    assert {"evidence_precision", "conditional_evidence_precision"} <= set(rec)
    summary = pd.read_csv(run.parents[1] / "summary.csv")
    assert "rec_evidence_precision_at_p50" in summary


def _with_calls(calls, unreviewed=None):
    """10 s file, two 5 s segments; calls: [(start, end, label)]."""
    seg = pd.DataFrame({"file": [F, F], "start": [0.0, 5.0], "end": [5.0, 10.0],
                        "labels": [sorted({l for s, e, l in calls if s < 5}), sorted({l for s, e, l in calls if e > 5})]})
    if unreviewed is not None:
        seg["unreviewed"] = unreviewed
    events = pd.DataFrame([(F, s, e, l) for s, e, l in calls], columns=["file", "start", "end", "label"])
    return EvalSet("c", seg, ["a", "b"], events=events)


def _evidence(ev, window, cls=0):
    units, _ = make_units(ev)
    sc = np.full((1, 2), -9.0, np.float32)
    sc[0, cls] = 5.0
    ws = WindowScores(F, np.array([window[0]]), np.array([window[1]]), sc, ["a", "b"])
    us = aggregate(ev, {F: ws}, units)
    return evidence_truth(ev, ev.target_matrix(), us)[0, cls]


def test_call_level_evidence_is_stricter_than_segment_labels():
    ev = _with_calls([(0.5, 1.5, "a")])  # segment 0-5 is labelled a
    # window 3-6 has 2 of its 3 s in segment 0-5: the segment check would accept it,
    # but it never overlaps the call
    assert _evidence(ev, (3.0, 6.0)) == 0
    assert _evidence(ev, (0.0, 3.0)) == 1
    seg_only = EvalSet("s", ev.segments, ev.classes)  # no events: segment fallback
    assert _evidence(seg_only, (3.0, 6.0)) == 1


def test_call_level_evidence_overlap_rule():
    ev = _with_calls([(2.0, 4.0, "a"), (8.0, 8.3, "a")])
    assert _evidence(ev, (3.6, 6.6)) == 0       # overlaps the 2 s call by 0.4 s < 0.5
    assert _evidence(ev, (3.5, 6.5)) == 1       # by 0.5 s
    assert _evidence(ev, (7.0, 10.0)) == 1      # a 0.3 s call wholly inside the window


def test_call_level_evidence_unknown_cases():
    # a is not reviewed in 5-10 s: a window there without an a call is unknown, not wrong
    ev = _with_calls([(0.5, 1.5, "a")], unreviewed=[[], ["a"]])
    assert _evidence(ev, (6.0, 9.0)) == -1
    # 5-10 s is unreviewed time (no species checked): a window reaching into it
    # without hitting a call is unknown; one that hits the call is still correct
    ev = _with_calls([(0.5, 1.5, "a")], unreviewed=[[], ["a", "b"]])
    assert _evidence(ev, (3.5, 6.5)) == -1
    assert _evidence(ev, (0.0, 3.0)) == 1

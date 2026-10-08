"""Phase 3: recording-level presence (units, unit truth, aggregation, metrics)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from birdeval import recording_metrics as rm
from birdeval.aggregate import aggregate, make_units, unit_truth
from birdeval.data import EvalSet, WindowScores
from birdeval.models.fake import FakeModel
from birdeval.runner import evaluate

from conftest import SPECIES


def _grid(files: dict[str, float], seg_s: float = 5.0) -> EvalSet:
    rows = [(f, k * seg_s, (k + 1) * seg_s, []) for f, dur in files.items() for k in range(int(dur // seg_s))]
    return EvalSet("g", pd.DataFrame(rows, columns=["file", "start", "end", "labels"]), ["a", "b"])


def test_unit_truth_three_states():
    # unit 0: a present in one segment; b all absent. unit 1: a unknown in one segment, else absent
    y = np.array([[0, 0], [1, 0], [0, 0], [-1, 0], [0, 0]], np.int8)
    seg_unit = np.array([0, 0, 1, 1, 1])
    np.testing.assert_array_equal(unit_truth(y, seg_unit, 2), [[1, 0], [-1, 0]])
    # a present segment beats an unknown one in the same unit
    np.testing.assert_array_equal(unit_truth(np.array([[1], [-1]], np.int8), np.array([0, 0]), 1), [[1]])


def test_unknown_units_are_excluded_from_metrics():
    T = np.array([[1], [0], [-1]], np.int8)
    S = np.array([[0.9], [0.1], [5.0]])  # the unknown unit has the top score
    r = rm.compute(T, S, np.full(3, 30.0), np.arange(3), targets=[0.99], n_boot=0)
    assert r["n_pairs"] == 2 and r["cmAP"] == 1.0
    assert r["targets"]["0.99"]["recall"] == 1.0


def test_unit_s_chunks_a_300s_file_into_five():
    ev = _grid({"/x/long.wav": 300.0, "/x/short.wav": 50.0})
    units, seg_unit = make_units(ev, unit_s=60)
    long_units = units[units.file == "/x/long.wav"]
    assert len(long_units) == 5 and long_units["duration_s"].tolist() == [60.0] * 5
    seg = ev.segments
    for u, row in long_units.iterrows():
        members = seg[seg_unit == u]
        assert len(members) == 12
        assert members.start.min() == row.start and members.end.max() == row.end
    assert len(units[units.file == "/x/short.wav"]) == 1  # 50 s: one partial chunk
    whole, seg_unit = make_units(ev)
    assert len(whole) == 2 and (np.bincount(seg_unit) == [60, 10]).all()


def test_aggregate_features_and_evidence():
    ev = _grid({"/x/f.wav": 20.0})
    units, _ = make_units(ev)
    sc = np.array([[1.0, -2.0], [3.0, -1.0], [2.0, -3.0], [-5.0, -0.5]], np.float32)
    ws = WindowScores("/x/f.wav", np.array([0, 5, 10, 15.0]), np.array([5, 10, 15, 20.0]), sc, ["a", "b"])
    us = aggregate(ev, {"/x/f.wav": ws}, units, tau=0.0)
    np.testing.assert_allclose(us.max_score, [[3.0, -0.5]])
    np.testing.assert_allclose(us.top3_mean, [[2.0, (-2 - 1 - 0.5) / 3]])
    np.testing.assert_array_equal(us.n_above, [[3, 0]])
    np.testing.assert_allclose(us.evidence_start, [[5.0, 15.0]])
    assert us.units.n_windows.tolist() == [4]


SCORES = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4])
TRUTH = np.array([1, 1, 0, 1, 0, 1])
# cut after k claims: precision 1, 1, 2/3, 3/4, 3/5, 4/6; recall (of 4) .25 .5 .5 .75 .75 1


@pytest.mark.parametrize(
    "target, thr, prec, rec, n",
    [(0.75, 0.6, 0.75, 0.75, 4), (0.9, 0.8, 1.0, 0.5, 2), (0.6, 0.4, 4 / 6, 1.0, 6), (0.5, 0.4, 4 / 6, 1.0, 6)],
)
def test_recall_at_precision_hand_computed(target, thr, prec, rec, n):
    r = rm.recall_at_precision(TRUTH, SCORES, target)
    assert r["reached"]
    assert (r["threshold"], r["recall"], r["n_claims"]) == (thr, rec, n)
    assert r["precision"] == pytest.approx(prec)


def test_recall_at_precision_prefers_higher_threshold_at_equal_recall():
    # cuts: precision 1, 1, 2/3; recall .5, 1, 1. Both later cuts clear 0.6.
    r = rm.recall_at_precision(np.array([1, 1, 0]), np.array([0.9, 0.8, 0.7]), 0.6)
    assert (r["threshold"], r["recall"], r["n_claims"]) == (0.8, 1.0, 2)


def test_recall_at_precision_unreachable_and_ties():
    r = rm.recall_at_precision(TRUTH, SCORES, 1.01)
    assert not r["reached"] and r["recall"] == 0.0 and np.isnan(r["threshold"])
    # a tie can't be split: claiming the top pair gives precision 0.5
    r = rm.recall_at_precision(np.array([1, 0, 1]), np.array([0.9, 0.9, 0.5]), 0.9)
    assert not r["reached"]


def test_list_precision_pooled_and_per_unit():
    truth = np.array([1, 0, 1, 1, 0, 0])
    score = np.array([0.9, 0.8, 0.1, 0.9, 0.9, 0.9])
    unit = np.array([0, 0, 0, 1, 1, 1])
    lp = rm.list_precision(truth, score, unit, threshold=0.5)
    assert lp["list_precision"] == pytest.approx(2 / 5)
    assert lp["list_precision_unit_mean"] == pytest.approx((1 / 2 + 1 / 3) / 2)
    assert lp["recall"] == pytest.approx(2 / 3)


def test_by_duration_uses_global_threshold():
    # short units: clean; long units: one absent species scores high
    T = np.array([[1, 0], [1, 0], [1, 0], [1, 0]], np.int8)
    S = np.array([[5, -5], [5, -5], [5, 6], [5, 6]], float)
    dur = np.array([30, 30, 1200, 1200.0])
    r = rm.compute(T, S, dur, np.arange(4), targets=[0.6], duration_bins_s=[60], n_boot=0)["targets"]["0.6"]
    assert r["threshold"] == 5
    short, long = r["by_duration"]
    assert short["list_precision"] == 1.0 and long["list_precision"] == 0.5


def test_runner_writes_recording_outputs(toy, tmp_path):
    _, tax, ev = toy
    m = FakeModel("oracle", tax, SPECIES, noise=0.0, window_s=5.0, hop_s=5.0)
    m.oracle = ev
    cfg = {"output_dir": str(tmp_path / "res"), "cache_dir": str(tmp_path / "cache"), "bootstrap": 10,
           "alignment": {"min_overlap_s": 0.1}}
    t = evaluate(cfg, [], [ev], run_name="r", models={"oracle": m})
    d = tmp_path / "res" / "r" / "toy" / "oracle"
    rec = json.loads((d / "recording_level.json").read_text())
    assert rec["all"]["n_units"] == 6
    pres = pd.read_csv(d / "presence.csv")
    assert len(pres) == 6 * len(ev.classes)
    assert {"unit", "species", "presence_conf", "n_above", "evidence_start", "evidence_end", "truth"} <= set(pres)
    row = t[t.classes == "all"].iloc[0]
    assert row.rec_cmAP == pytest.approx(1.0) and row.rec_recall_at_p90 == pytest.approx(1.0)
    # disabled: no recording-level files or columns
    t2 = evaluate({**cfg, "recording_level": {"enabled": False}}, [], [ev], run_name="off", models={"oracle": m})
    assert "rec_cmAP" not in t2 and not (tmp_path / "res" / "off" / "toy" / "oracle" / "presence.csv").exists()

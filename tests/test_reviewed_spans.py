"""Recording-level units are trimmed to reviewed time, so an unreviewed stretch
no longer makes every absent species unknown for the whole recording."""

from __future__ import annotations

import numpy as np
import pandas as pd

from birdeval.aggregate import aggregate, make_units, unit_truth
from birdeval.data import EvalSet, WindowScores

CLASSES = ["a", "b"]
ALL = list(CLASSES)


def _ev(rows):
    return EvalSet("r", pd.DataFrame(rows, columns=["file", "start", "end", "labels", "unreviewed"]), CLASSES)


def test_unreviewed_tail_is_trimmed_not_unknown():
    # reviewed 0-10 s (a present), then a 2 s tail no labelled clip covers
    ev = _ev([("f", 0.0, 5.0, ["a"], []), ("f", 5.0, 10.0, [], []), ("f", 10.0, 12.0, [], ALL)])
    units, seg_unit = make_units(ev)
    assert seg_unit.tolist() == [0, 0, -1]
    assert units.spans[0] == [(0.0, 10.0)]
    assert units.duration_s.tolist() == [10.0]
    T = unit_truth(ev.target_matrix(), seg_unit, len(units))
    # b is absent from all reviewed time: 0, not -1 as it was before trimming
    np.testing.assert_array_equal(T, [[1, 0]])


def test_interior_gap_splits_spans_and_windows_in_it_dont_count():
    ev = _ev([("f", 0.0, 5.0, [], []), ("f", 5.0, 10.0, [], ALL), ("f", 10.0, 15.0, [], [])])
    units, _ = make_units(ev)
    assert units.spans[0] == [(0.0, 5.0), (10.0, 15.0)] and units.duration_s[0] == 10.0
    s = np.arange(0, 12.5, 2.5)  # 5 s windows, 2.5 s hop
    ws = WindowScores("f", s, s + 5, np.arange(len(s), dtype=np.float32)[:, None].repeat(2, 1), CLASSES)
    us = aggregate(ev, {"f": ws}, units)
    # [0,5] and [10,15] count; [2.5,7.5] / [7.5,12.5] are only half in reviewed time; [5,10] is all in the gap
    assert us.units.n_windows.tolist() == [2]
    assert us.max_score[0, 0] == 4.0 and us.evidence_start[0, 0] == 10.0


def test_species_level_review_gaps_still_unknown():
    ev = _ev([("f", 0.0, 5.0, [], ["b"]), ("f", 5.0, 10.0, [], [])])
    units, seg_unit = make_units(ev)
    assert len(units) == 1 and units.duration_s[0] == 10.0  # time is reviewed, species b isn't
    np.testing.assert_array_equal(unit_truth(ev.target_matrix(), seg_unit, 1), [[0, -1]])


def test_fully_unreviewed_file_has_no_unit():
    ev = _ev([("f", 0.0, 5.0, [], []), ("g", 0.0, 5.0, [], ALL)])
    units, seg_unit = make_units(ev)
    assert units.file.tolist() == ["f"] and seg_unit.tolist() == [0, -1]


def test_chunks_skip_unreviewed_time():
    rows = [("f", k * 5.0, k * 5.0 + 5, [], ALL if 12 <= k < 24 else []) for k in range(36)]  # 180 s, 60-120 s unreviewed
    units, _ = make_units(_ev(rows), unit_s=60)
    assert units.start.tolist() == [0.0, 120.0] and units.duration_s.tolist() == [60.0, 60.0]

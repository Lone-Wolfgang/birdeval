"""The half-window alignment rule: a window counts towards a span only if more
than half of it (strictly) lies inside, measured on the window's own length."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from birdeval.aggregate import aggregate, make_units
from birdeval.align import window_hits
from birdeval.data import EvalSet, WindowScores


def _windows(length: float, hop: float, total: float):
    starts = np.arange(0, total - 1e-9, hop)
    return starts, np.minimum(starts + length, total)  # last windows may be partial


def test_perch_geometry_counts_only_the_centred_window():
    s, e = _windows(5.0, 2.5, 20.0)  # neighbours overlap each 5 s segment by exactly 2.5 s
    hit = window_hits(s, e, [(5.0, 10.0)])
    assert s[hit].tolist() == [5.0]  # 2.5 s is exactly half: strict > excludes it


def test_birdnet_geometry():
    s, e = _windows(3.0, 1.5, 20.0)
    hit = window_hits(s, e, [(5.0, 10.0)])
    # overlaps: [3,6] 1 s, [4.5,7.5] 2.5, [6,9] 3, [7.5,10.5] 2.5, [9,12] 1
    assert s[hit].tolist() == [4.5, 6.0, 7.5]


@pytest.mark.parametrize("length, hop", [(5.0, 2.5), (3.0, 1.5), (3.0, 1.0), (5.0, 1.0)])
def test_no_window_counts_towards_two_segments(length, hop):
    s, e = _windows(length, hop, 60.0)
    grid = [(k * 5.0, k * 5.0 + 5) for k in range(12)]
    counts = sum(window_hits(s, e, [g]).astype(int) for g in grid)
    assert counts.max() == 1


def test_partial_window_measured_on_its_own_length():
    s, e = np.array([9.0]), np.array([10.0])  # 1 s tail window at the end of a file
    assert window_hits(s, e, [(5.0, 10.0)]).tolist() == [True]


def test_short_segment_falls_back_to_best_window():
    s, e = _windows(3.0, 1.5, 6.0)
    # a 1 s segment: no 3 s window can have > 1.5 s inside
    hit = window_hits(s, e, [(0.0, 1.0)])
    assert s[hit].tolist() == [0.0]


def test_spans_add_up_and_bad_rule_raises():
    s, e = np.array([4.0]), np.array([7.0])
    # 1 s in each of two spans = 2 s of 3 s -> counts
    assert window_hits(np.array([4.0, 20.0]), np.array([7.0, 23.0]), [(4.0, 5.0), (6.0, 7.0)]).tolist() == [True, False]
    with pytest.raises(ValueError, match="unknown alignment rule"):
        window_hits(s, e, [(0, 5)], rule="centre")


def test_unit_membership_uses_the_same_rule():
    rows = [("/x/f.wav", k * 5.0, k * 5.0 + 5, []) for k in range(2)]  # a 10 s unit
    ev = EvalSet("e", pd.DataFrame(rows, columns=["file", "start", "end", "labels"]), ["a"])
    s, e = _windows(5.0, 2.5, 20.0)
    ws = WindowScores("/x/f.wav", s, e, np.arange(len(s), dtype=np.float32)[:, None], ["a"])
    units, _ = make_units(ev)
    us = aggregate(ev, {"/x/f.wav": ws}, units)
    # [0,5], [2.5,7.5], [5,10] count; [7.5,12.5] has exactly half inside and doesn't
    assert us.units.n_windows.tolist() == [3]
    assert us.max_score[0, 0] == 2.0 and us.evidence_start[0, 0] == 5.0

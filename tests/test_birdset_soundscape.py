"""BirdSet soundscape mode, bootstrap groups, the finite floor, and per-species
precision. The HF download is stubbed: no network or BirdSet data needed."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from birdeval import recording_metrics as rm
from birdeval import scoring
from birdeval.aggregate import aggregate, make_units
from birdeval.align import align
from birdeval.data import EvalSet, WindowScores
from birdeval.evalsets import birdset
from birdeval.models.fake import FakeModel
from birdeval.runner import evaluate

from conftest import SPECIES


def test_parse_clip_name():
    assert birdset.parse_clip_name("/x/Recording_1_Segment_03_005_010.ogg") == ("Recording_1_Segment_03.ogg", 5.0)
    assert birdset.parse_clip_name("/x/UHH_001_S01_20161121_150000_295_300.ogg") == (
        "UHH_001_S01_20161121_150000.ogg", 295.0)
    assert birdset.parse_clip_name("/x/no_offsets.ogg") is None


def _clips(names_lengths_labels):
    return pd.DataFrame([(f"/dl/{n}", 0.0, l, lab) for n, l, lab in names_lengths_labels],
                        columns=["file", "start", "end", "labels"])


def test_clips_land_at_their_offsets_with_true_length():
    clips = _clips([("A_005_010.ogg", 5.0, ["b"]), ("A_000_005.ogg", 5.0, ["a"]), ("A_010_015.ogg", 1.5, [])])
    seg = birdset.clips_to_soundscape(clips, {"A.ogg": "/s/A.ogg"})
    assert seg[["start", "end"]].values.tolist() == [[0, 5], [5, 10], [10, 11.5]]  # partial last clip
    assert seg.labels.tolist() == [["a"], ["b"], []]
    with pytest.raises(ValueError, match="not in the test split"):
        birdset.clips_to_soundscape(clips, {})
    with pytest.raises(ValueError, match="don't match"):
        birdset.clips_to_soundscape(_clips([("bad.ogg", 5.0, [])]), {})


def test_uncovered_time_becomes_unreviewed_segments():
    seg = pd.DataFrame({"file": ["f"] * 2, "start": [5.0, 10.0], "end": [10.0, 11.5], "labels": [[], []]})
    gaps = birdset.uncovered_segments(seg, {"f": 18.0}, ["a", "b"])
    # 0-5 before the first clip; 11.5-18 after the last, cut into <= 5 s pieces
    assert gaps[["start", "end"]].values.tolist() == [[0, 5], [11.5, 16.5], [16.5, 18]]
    assert all(u == ["a", "b"] for u in gaps.unreviewed)
    # float slack at the end is not a gap
    covers_0_10 = seg.iloc[[0]].assign(start=0.0)
    assert birdset.uncovered_segments(covers_0_10, {"f": 10.004}, ["a"]).empty


@pytest.fixture
def stub_birdset(tmp_path, monkeypatch):
    """Two soundscapes: A (12 s; clips cover 0-10) and B (10 s, fully covered)."""
    paths = {}
    for n, dur in (("A.ogg", 12.0), ("B.ogg", 10.0)):
        sf.write(tmp_path / n, np.zeros(int(16_000 * dur), np.float32), 16_000)
        paths[n] = str((tmp_path / n).resolve())
    clips = _clips([("A_000_005.ogg", 5.0, ["x"]), ("A_005_010.ogg", 5.0, []),
                    ("B_000_005.ogg", 5.0, []), ("B_005_010.ogg", 5.0, ["y"])])

    class _Scapes:
        def unique(self, col):
            return list(paths.values())

    monkeypatch.setattr(birdset, "_clip_table", lambda *a: (clips, ["x", "y"], []))
    monkeypatch.setattr(birdset, "_load", lambda *a: _Scapes())
    return paths


def test_soundscape_loader_marks_uncovered_time(stub_birdset):
    ev = birdset.load_birdset_soundscapes("birdset_T_scape", None, "T")
    a = ev.segments[ev.segments.file == stub_birdset["A.ogg"]]
    assert a[["start", "end"]].values.tolist() == [[0, 5], [5, 10], [10, 12]]
    y = ev.target_matrix()
    np.testing.assert_array_equal(y[a.index], [[1, 0], [0, 0], [-1, -1]])
    assert ev.meta["uncovered_s"] == pytest.approx(2.0) and ev.meta["files_with_uncovered_time"] == 1
    # the source recording is the file: bootstrap groups are whole soundscapes
    assert set(ev.groups()) == set(stub_birdset.values())


def test_clip_mode_groups_clips_by_soundscape():
    seg = pd.DataFrame({"file": ["/c/A_000_005.ogg", "/c/A_005_010.ogg", "/c/B_000_005.ogg"],
                        "start": 0.0, "end": 5.0, "labels": [[], [], []],
                        "recording": ["A.ogg", "A.ogg", "B.ogg"]})
    ev = EvalSet("clips", seg, ["a"])
    assert ev.groups().tolist() == ["A.ogg", "A.ogg", "B.ogg"]
    units, _ = make_units(ev)
    assert units.group.tolist() == ["A.ogg", "A.ogg", "B.ogg"]  # 3 clip units, 2 bootstrap groups


def test_chunked_units_resample_by_source_recording():
    rows = [(f, k * 5.0, k * 5.0 + 5, []) for f in ("/x/r1.wav", "/x/r2.wav") for k in range(24)]  # 120 s each
    ev = EvalSet("e", pd.DataFrame(rows, columns=["file", "start", "end", "labels"]), ["a"])
    units, _ = make_units(ev, unit_s=60)
    assert len(units) == 4
    assert units.group.tolist() == ["/x/r1.wav", "/x/r1.wav", "/x/r2.wav", "/x/r2.wav"]


def test_unscoreable_classes_get_a_finite_floor(toy):
    _, _, ev = toy
    ws = {f: WindowScores(f, np.array([0.0]), np.array([5.0]), np.array([[2.0]], np.float32), ["turdus merula"])
          for f in ev.files}
    floor = scoring.floor("logit")
    assert np.isfinite(floor) and floor < -1e5
    s, covered = align(ev, ws, min_overlap_s=1.0, missing_value=floor)
    assert np.isfinite(s).all()
    other = [i for i, c in enumerate(ev.classes) if c != "turdus merula"]
    assert (s[:, other] == floor).all() and not covered[other].any()
    # segments no window reaches also get the floor, not 0 (= p 0.5 in logit space)
    tm = ev.classes.index("turdus merula")
    late = (ev.segments.start >= 10).to_numpy()
    assert (s[late, tm] == floor).all()
    us = aggregate(ev, ws, make_units(ev)[0], floor=floor)
    assert np.isfinite(us.max_score).all() and (us.max_score[:, other] == floor).all()


def test_per_species_precision_at_global_threshold():
    T = np.array([[1, 0], [1, 1], [0, 0], [-1, 0]], np.int8)
    S = np.array([[3.0, 2.5], [1.0, 2.0], [2.0, 0.0], [9.0, 9.0]])
    df = rm.per_species(T, S, ["a", "b"], np.arange(2), {"0.9": 2.0}).set_index("species")
    # a: claims at >= 2 on known units: units 0 (present) and 2 (absent)
    assert df.loc["a", "claims_at_p90"] == 2 and df.loc["a", "precision_at_p90"] == 0.5
    assert df.loc["a", "recall_at_p90"] == 0.5 and df.loc["a", "n_units"] == 3
    # b: claims units 0 (absent) and 1 (present); unit 3 is unknown for a only
    assert df.loc["b", "claims_at_p90"] == 3 and df.loc["b", "precision_at_p90"] == pytest.approx(1 / 3)


def test_runner_writes_per_species_and_checks_operating_target(toy, tmp_path):
    _, tax, ev = toy
    m = FakeModel("noisy", tax, SPECIES, noise=0.6, seed=5)
    m.oracle = ev
    cfg = {"output_dir": str(tmp_path / "res"), "cache_dir": str(tmp_path / "cache"), "bootstrap": 0}
    t = evaluate(cfg, [], [ev], run_name="r", models={"noisy": m})
    assert {"rec_recall_at_p80", "rec_recall_at_p90", "rec_recall_at_p95"} <= set(t.columns)
    ps = pd.read_csv(tmp_path / "res" / "r" / "toy" / "noisy" / "per_species_recording_all.csv")
    assert {"precision_at_p80", "precision_at_p95", "recall_at_p90"} <= set(ps.columns)
    with pytest.raises(ValueError, match="operating_target"):
        evaluate({**cfg, "recording_level": {"operating_target": 0.7}}, [], [ev], run_name="x", models={"noisy": m})

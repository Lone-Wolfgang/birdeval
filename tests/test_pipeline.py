"""End-to-end test with synthetic audio + a fake model. No weights or downloads."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from birdeval.align import align
from birdeval.data import WindowScores
from birdeval.evalsets.strong_labels import load_strong_labels
from birdeval.models.fake import FakeModel
from birdeval.runner import evaluate
from birdeval.taxonomy import Taxonomy

SPECIES = ["Turdus merula", "Erithacus rubecula", "Parus major", "Fringilla coelebs"]


@pytest.fixture
def toy(tmp_path):
    rng = np.random.default_rng(1)
    rows = []
    for i in range(6):
        f = tmp_path / f"rec{i}.wav"
        sf.write(f, np.zeros(32_000 * 30, np.float32), 32_000)  # 30 s silence
        for _ in range(5):
            s = float(rng.uniform(0, 27))
            rows.append((f.name, s, s + float(rng.uniform(0.3, 3)), SPECIES[rng.integers(0, 3)]))
    ann = tmp_path / "ann.csv"
    pd.DataFrame(rows, columns=["file", "start", "end", "label"]).to_csv(ann, index=False)
    tax = Taxonomy()
    ev = load_strong_labels("toy", tax, str(ann), audio_root=str(tmp_path))
    return tmp_path, tax, ev


def test_taxonomy_normalisation():
    t = Taxonomy()
    assert t.from_birdnet("Turdus merula_Eurasian Blackbird") == "turdus merula"
    assert t.from_sci("  Turdus   Merula ") == "turdus merula"


def test_evalset_grid(toy):
    _, _, ev = toy
    assert len(ev.segments) == 6 * 6  # 30 s / 5 s
    assert ev.target_matrix().sum() > 0


def test_align_max_pools_overlapping_windows(toy):
    _, _, ev = toy
    f = ev.files[0]
    ws = WindowScores(
        f, np.array([0.0, 4.0]), np.array([3.0, 7.0]),
        np.array([[0.2], [0.9]], np.float32), ["turdus merula"],
    )
    s, covered = align(ev, {f: ws} | {g: WindowScores(g, np.zeros(0), np.zeros(0), np.zeros((0, 0)), []) for g in ev.files[1:]})
    col = ev.classes.index("turdus merula")
    first = ev.segments.index[(ev.segments.file == f) & (ev.segments.start == 0)][0]
    assert s[first, col] == pytest.approx(0.9)  # [0,5] overlaps [4,7] by 1 s -> included
    assert covered[col]


def test_perfect_model_scores_one(toy, tmp_path):
    root, tax, ev = toy
    # windows aligned to the 5 s grid; overlapping windows would bleed into neighbours
    m = FakeModel("oracle", tax, SPECIES, noise=0.0, window_s=5.0, hop_s=5.0)
    m.oracle = ev
    cfg = {"output_dir": str(tmp_path / "res"), "cache_dir": str(tmp_path / "cache"), "bootstrap": 20,
           "alignment": {"min_overlap_s": 0.1}}
    t = evaluate(cfg, ["oracle"], [ev], run_name="t", models={"oracle": m})
    row = t[(t.classes == "all")].iloc[0]
    assert row.cmAP == pytest.approx(1.0)
    assert row.top1 == pytest.approx(1.0)


def test_noisy_vs_partial_vocab_and_cache(toy, tmp_path):
    root, tax, ev = toy
    good = FakeModel("good", tax, SPECIES, noise=0.6, seed=1)
    partial = FakeModel("partial", tax, SPECIES[:2], noise=0.6, seed=2)  # can't predict Parus major
    for m in (good, partial):
        m.oracle = ev
    cfg = {"output_dir": str(tmp_path / "res"), "cache_dir": str(tmp_path / "cache"), "bootstrap": 20}
    t = evaluate(cfg, [], [ev], run_name="a", models={"good": good, "partial": partial})
    s = t.set_index(["model", "classes"])
    assert s.loc[("partial", "shared"), "n_classes_scored"] == 2
    assert s.loc[("good", "all"), "n_classes_scored"] == 3
    # second run must come entirely from cache and give identical numbers
    good.predict_files = lambda *a, **k: (_ for _ in ()).throw(AssertionError("cache miss"))
    t2 = evaluate(cfg, [], [ev], run_name="b", models={"good": good, "partial": partial})
    pd.testing.assert_frame_equal(t.reset_index(drop=True), t2.reset_index(drop=True))

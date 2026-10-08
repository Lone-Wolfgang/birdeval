"""Interrupted runs keep the files they finished, and resume from there."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from birdeval.cache import PredictionCache
from birdeval.data import WindowScores
from birdeval.models.birdnet_lib import BirdNETLibModel
from birdeval.models.fake import FakeModel
from birdeval.runner import evaluate
from birdeval.scoring import sigmoid
from birdeval.taxonomy import Taxonomy

from conftest import SPECIES


def crash_after(model: FakeModel, n: int) -> FakeModel:
    """Make `model` stream its first n files, then die like a killed run.

    Patched on the instance: a subclass would change the cache key (it hashes
    the adapter class), and the restart has to see the same cache.
    """
    real = model.predict_files

    def predict_files(files, vocab, on_result=None):
        real(files[:n], vocab, on_result)
        raise RuntimeError("simulated crash")

    model.predict_files = predict_files
    return model


def test_interrupted_run_resumes_from_cached_files(toy, tmp_path):
    _, tax, ev = toy
    cfg = {"output_dir": str(tmp_path / "res"), "cache_dir": str(tmp_path / "cache"), "bootstrap": 0}
    with pytest.raises(RuntimeError, match="simulated crash"):
        evaluate(cfg, [], [ev], run_name="a", models={"m": crash_after(FakeModel("m", tax, SPECIES, noise=0.5), 3)})

    # a fresh process with the same config sees the 3 finished files
    model = FakeModel("m", tax, SPECIES, noise=0.5)
    cache = PredictionCache(cfg["cache_dir"], model.cache_key(), ev.name, ev.classes)
    finished = [f for f in ev.files if cache.has(f)]
    assert len(finished) == 3
    before = {f: cache.get(f).scores.copy() for f in finished}

    seen = []
    real = model.predict_files
    model.predict_files = lambda files, vocab, on_result=None: seen.extend(files) or real(files, vocab, on_result)
    evaluate(cfg, [], [ev], run_name="b", models={"m": model})
    assert sorted(seen) == sorted(set(ev.files) - set(finished))  # only the rest is re-run
    for f in finished:  # and the finished files were not overwritten
        np.testing.assert_array_equal(cache.get(f).scores, before[f])


def test_cache_write_is_all_or_nothing(tmp_path, monkeypatch):
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"x")
    cache = PredictionCache(tmp_path / "cache", "k", "e", ["a"])
    ws = WindowScores(str(audio), np.zeros(1), np.ones(1), np.zeros((1, 1), np.float32), ["a"])

    def dies_mid_write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"PK\x03\x04 truncated")  # a half-written npz
        raise OSError("killed mid-write")

    monkeypatch.setattr(WindowScores, "save", dies_mid_write)
    with pytest.raises(OSError):
        cache.put(ws)
    assert not cache.has(str(audio)) and cache.missing([str(audio)]) == [str(audio)]
    assert not list(cache.dir.glob("*.tmp.npz"))  # no stray temp files

    monkeypatch.undo()
    cache.put(ws)
    assert cache.has(str(audio)) and cache.get(str(audio)).classes == ["a"]


def _one_file_result(path: str, probs: list[float]):
    """A birdnet-shaped result for one file, one window, full species list A, B."""
    return SimpleNamespace(
        species_list=np.array(["A a_Aa", "B b_Bb"]),
        species_ids=np.array([[[0, 1]]], np.uint16),
        species_probs=np.array([[probs]], np.float32),
        species_masked=np.array([[[False, False]]]),
        hop_duration_s=3.0, segment_duration_s=3.0,
        input_durations=np.array([3.0]), inputs=np.array([path]),
    )


class _FakeBirdnet:
    species_list = ["A a_Aa", "B b_Bb"]

    def __init__(self, probs):
        self.probs, self.calls = probs, []

    def predict(self, files, **kw):
        self.calls.append(kw)
        for f in files:  # birdnet streams each file before returning the batch
            if "on_file_complete" in kw:
                kw["on_file_complete"](_one_file_result(f, self.probs))
        return SimpleNamespace(**{**vars(_one_file_result(files[0], self.probs)), "inputs": np.array(files[:1])})


@pytest.mark.parametrize("version", ["2.4", "3.0"])
def test_birdnet_adapter_streams_each_file(tmp_path, version):
    m = BirdNETLibModel("m", Taxonomy(), version=version)
    m._model = _FakeBirdnet([0.8, 0.3])
    files = [str((tmp_path / f"{i}.wav").resolve()) for i in range(3)]
    got = []
    m.predict_files(files, {"a a", "b b"}, on_result=got.append)
    assert [w.file for w in got] == files
    assert "on_file_complete" in m.model.calls[0]
    s = got[0].scores
    if version == "3.0":  # probabilities -> reconstructed logits, same as the batch path
        np.testing.assert_allclose(sigmoid(s), [[0.8, 0.3]], rtol=1e-5)
    else:
        np.testing.assert_allclose(s, [[0.8, 0.3]])
    # without a callback nothing extra is passed to birdnet
    m.predict_files(files, {"a a", "b b"})
    assert "on_file_complete" not in m.model.calls[-1]

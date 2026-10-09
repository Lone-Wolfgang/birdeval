"""Phase 1: raw logits in the cache, and the scorer stage."""

from __future__ import annotations

import numpy as np
import pytest

from birdeval import scoring
from birdeval.data import WindowScores
from birdeval.models.birdnet_lib import BirdNETLibModel, prob_to_logit
from birdeval.models.fake import FakeModel
from birdeval.runner import evaluate
from birdeval.taxonomy import Taxonomy

from conftest import SPECIES


def test_sigmoid_is_stable_and_unclipped():
    x = np.array([-800.0, -20.0, 0.0, 16.0, 20.0, 800.0])
    p = scoring.sigmoid(x)
    assert np.isfinite(p).all()
    assert p[2] == 0.5
    # birdnet clipped at 15 and float32 saturates ~17: both would tie these
    assert p[3] < p[4] < 1.0


def test_logit_is_identity_and_unknown_method_raises():
    ws = WindowScores("f", np.zeros(1), np.ones(1), np.array([[3.0]], np.float32), ["a"])
    assert scoring.apply(ws, "logit") is ws
    with pytest.raises(ValueError):
        scoring.apply(ws, "softmax")


def test_prob_to_logit_round_trips():
    p = np.concatenate([[1e-6, 1 - 1e-6], np.linspace(1e-6, 1 - 1e-6, 1001)])
    np.testing.assert_allclose(scoring.sigmoid(prob_to_logit(p)), p, rtol=1e-5, atol=1e-9)
    assert np.isfinite(prob_to_logit(np.array([0.0, 1.0]))).all()


def test_score_space_changes_cache_key():
    tax = Taxonomy()
    m = BirdNETLibModel("m", tax, version="2.4")
    assert m.version_info()["score_space"] == "logit"
    assert BirdNETLibModel("m", tax, version="3.0").version_info()["score_space"] == "logit_from_prob"
    old = BirdNETLibModel("m", tax, version="2.4")
    del old.params["score_space"]  # what a pre-logit adapter hashed
    assert m.cache_key() != old.cache_key()


def test_synonyms_change_cache_key(tmp_path):
    # A synonym makes more native labels count for the eval classes, so the
    # cached logits must not be reused.
    syn = tmp_path / "synonyms.yaml"
    syn.write_text('"Dryobates villosus": "Leuconotopicus villosus"\n')
    plain = BirdNETLibModel("m", Taxonomy(), version="2.4")
    mapped = BirdNETLibModel("m", Taxonomy(synonyms_yaml=syn), version="2.4")
    assert plain.cache_key() != mapped.cache_key()
    assert "synonyms" not in plain.version_info()

def test_logit_and_sigmoid_give_identical_ranking_metrics(toy, tmp_path):
    _, tax, ev = toy
    m = FakeModel("noisy", tax, SPECIES, noise=0.6, seed=3)
    m.oracle = ev
    base = {"output_dir": str(tmp_path / "res"), "cache_dir": str(tmp_path / "cache"), "bootstrap": 0}
    a = evaluate({**base, "scoring": {"method": "logit"}}, [], [ev], run_name="l", models={"noisy": m})
    # second run must not re-run inference: the scorer works on cached logits
    m.predict_files = lambda *a, **k: (_ for _ in ()).throw(AssertionError("cache miss"))
    b = evaluate({**base, "scoring": {"method": "sigmoid"}}, [], [ev], run_name="s", models={"noisy": m})
    cols = ["cmAP", "AUROC", "top1"]
    np.testing.assert_allclose(a[cols].to_numpy(), b[cols].to_numpy(), rtol=1e-12)

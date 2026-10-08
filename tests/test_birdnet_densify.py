"""Checks the birdnet-result -> dense matrix conversion against a stand-in object
shaped like birdnet 1.1.1's AcousticPredictionResult (ids index the FULL species
list; non-listed species are masked; padding segments are fully masked)."""

from types import SimpleNamespace

import numpy as np
from pathlib import Path

from birdeval.models.birdnet_lib import _densify


def test_densify_maps_ids_through_full_species_list():
    full = np.array(["A a_Aa", "B b_Bb", "C c_Cc", "D d_Dd"])
    native = ["D d_Dd", "B b_Bb"]  # restricted, in a different order
    canon = ["d d", "b b"]
    # 1 file, 3 segment slots (last is padding), k=2, sorted by prob desc
    ids = np.array([[[3, 1], [1, 3], [0, 0]]], dtype=np.uint16)
    probs = np.array([[[0.9, 0.2], [0.7, 0.1], [0, 0]]], dtype=np.float32)
    masked = np.array([[[False, False], [False, False], [True, True]]])
    res = SimpleNamespace(
        species_list=full, species_ids=ids, species_probs=probs, species_masked=masked,
        hop_duration_s=1.5, segment_duration_s=3.0, input_durations=np.array([4.0]),
        inputs=np.array(["/tmp/x.wav"]),
    )
    out = _densify(res, native, canon)[str(Path("/tmp/x.wav").resolve())]
    assert out.classes == canon
    np.testing.assert_allclose(out.scores, [[0.9, 0.2], [0.1, 0.7]])
    np.testing.assert_allclose(out.starts, [0.0, 1.5])
    np.testing.assert_allclose(out.ends, [3.0, 4.0])  # clipped to file duration

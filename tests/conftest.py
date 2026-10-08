"""Shared fixtures: synthetic audio + strong labels. No weights or downloads."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from birdeval.evalsets.strong_labels import load_strong_labels
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

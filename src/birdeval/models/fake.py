"""A deterministic fake model for testing the pipeline without weights.

It "cheats" by reading the eval set's ground truth (passed in as `oracle`)
and adding noise, so metrics land in a known range: noise=0 should give
cmAP == 1.0.
"""

from __future__ import annotations

import numpy as np
import soundfile as sf

from ..data import WindowScores
from ..taxonomy import Taxonomy
from .base import ModelAdapter


class FakeModel(ModelAdapter):
    def __init__(
        self,
        name: str,
        taxonomy: Taxonomy,
        classes: list[str],
        window_s: float = 3.0,
        hop_s: float = 1.5,
        noise: float = 0.3,
        seed: int = 0,
    ) -> None:
        super().__init__(name, taxonomy, window_s=window_s, hop_s=hop_s, noise=noise, seed=seed)
        self._classes = classes
        self.window_s, self.hop_s, self.noise, self.seed = window_s, hop_s, noise, seed
        self.oracle = None  # set by tests: EvalSet

    def labels(self) -> list[str]:
        return self._classes

    def to_canonical(self, label: str) -> str | None:
        return self.taxonomy.from_sci(label)

    def predict_files(self, files, vocab, on_result=None):
        rng = np.random.default_rng(self.seed)
        cls = [c for c in map(self.to_canonical, self._classes) if c in vocab]
        out = {}
        for f in files:
            dur = sf.info(f).duration
            starts = np.arange(0, max(dur - self.window_s, 0) + 1e-9, self.hop_s)
            ends = np.minimum(starts + self.window_s, dur)
            s = rng.random((len(starts), len(cls))).astype(np.float32) * self.noise
            if self.oracle is not None:
                seg = self.oracle.segments[self.oracle.segments["file"] == f]
                for _, r in seg.iterrows():
                    hit = (np.minimum(ends, r.end) - np.maximum(starts, r.start)) > 0
                    for l in r.labels:
                        if l in cls:
                            s[hit, cls.index(l)] += 1.0 - self.noise
            out[f] = WindowScores(f, starts, ends, np.clip(s, 0, 1), cls)
            if on_result is not None:
                on_result(out[f])
        return out

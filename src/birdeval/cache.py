"""On-disk cache of raw window-level predictions.

Layout: <root>/<model cache key>/<evalset>-<vocab hash>/<file hash>.npz

Inference is the expensive step, so it is cached *before* alignment: changing
the alignment rule or adding a metric never requires re-running a model. The
model cache key hashes every parameter that changes outputs (version, backend,
overlap, ...), so changing one starts a fresh cache instead of mixing results.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

from .data import WindowScores


def _h(s: str, n: int = 12) -> str:
    return hashlib.sha1(s.encode()).hexdigest()[:n]


class PredictionCache:
    def __init__(self, root: str | Path, model_key: str, evalset: str, vocab: list[str]) -> None:
        self.dir = Path(root) / model_key / f"{evalset}-{_h('|'.join(sorted(vocab)), 8)}"

    def _path(self, file: str) -> Path:
        st = os.stat(file)
        return self.dir / f"{_h(f'{file}|{st.st_size}|{int(st.st_mtime)}', 16)}.npz"

    def get(self, file: str) -> WindowScores | None:
        p = self._path(file)
        return WindowScores.load(p) if p.exists() else None

    def put(self, ws: WindowScores) -> None:
        ws.save(self._path(ws.file))

    def missing(self, files: list[str]) -> list[str]:
        return [f for f in files if not self._path(f).exists()]

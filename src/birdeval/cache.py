"""On-disk cache of raw window-level predictions.

Layout: <root>/<model cache key>/<evalset>-<vocab hash>/<file hash>.npz

Inference is the expensive step, so it is cached *before* alignment: changing
the alignment rule or adding a metric never requires re-running a model. The
model cache key hashes every parameter that changes outputs (version, backend,
overlap, ...), so changing one starts a fresh cache instead of mixing results.

Interrupted runs resume: adapters hand each file to the cache as soon as it
is scored, and every write is atomic (temp file + rename), so a file is
either fully cached or missing, never half-written.
"""

from __future__ import annotations

import hashlib
import os
import threading
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
        path = self._path(ws.file)
        # unique per writer; ends in .npz so numpy doesn't append another suffix
        tmp = path.with_name(f"{path.stem}.{os.getpid()}-{threading.get_ident()}.tmp.npz")
        try:
            ws.save(tmp)
            os.replace(tmp, path)  # atomic on POSIX: readers see all or nothing
        finally:
            tmp.unlink(missing_ok=True)

    def has(self, file: str) -> bool:
        return self._path(file).exists()

    def missing(self, files: list[str]) -> list[str]:
        return [f for f in files if not self.has(f)]

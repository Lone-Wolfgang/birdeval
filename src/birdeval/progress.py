"""Per-file inference progress for one model on one eval set.

In a terminal: a live tqdm bar with rate and ETA. Otherwise (output going to
a log file): an INFO line every `every_pct` percent, since a redrawn bar is
unreadable in a file. Updates may come from another thread (birdnet's
file-completion dispatcher), so counting is locked.
"""

from __future__ import annotations

import logging
import sys
import threading
import time
from datetime import timedelta

log = logging.getLogger("birdeval")


def _hms(seconds: float) -> str:
    return str(timedelta(seconds=round(seconds)))


class Progress:
    def __init__(self, total: int, desc: str, every_pct: float = 10.0, live: bool | None = None) -> None:
        self.total, self.desc, self.every = total, desc, every_pct
        self.n, self._next = 0, every_pct
        self._lock = threading.Lock()
        self._t0 = time.time()
        live = sys.stderr.isatty() if live is None else live
        self._bar = None
        if live:
            from tqdm import tqdm

            self._bar = tqdm(total=total, desc=desc, unit="file", dynamic_ncols=True)

    def update(self, n: int = 1) -> None:
        with self._lock:
            self.n = min(self.n + n, self.total)
            if self._bar is not None:
                self._bar.update(n)
                return
            pct = 100.0 * self.n / max(self.total, 1)
            if pct + 1e-9 < self._next and self.n < self.total:
                return
            while self._next <= pct + 1e-9:
                self._next += self.every
            elapsed = time.time() - self._t0
            left = elapsed * (self.total - self.n) / self.n if self.n else float("nan")
            log.info("%s: %d/%d files (%.0f%%), %s elapsed, ~%s left",
                     self.desc, self.n, self.total, pct, _hms(elapsed), _hms(left) if self.n else "?")

    def close(self) -> None:
        if self._bar is not None:
            self._bar.close()

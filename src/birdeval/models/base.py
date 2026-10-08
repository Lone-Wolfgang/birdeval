from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod

from ..data import WindowScores
from ..taxonomy import Taxonomy


class ModelAdapter(ABC):
    """Wraps one model behind a single call.

    Subclasses implement `labels()` (native label strings), `to_canonical()`
    and `predict_files()`. Everything else (caching, alignment, metrics) is
    shared.
    """

    def __init__(self, name: str, taxonomy: Taxonomy, **params) -> None:
        self.name = name
        self.taxonomy = taxonomy
        self.params = params

    @abstractmethod
    def labels(self) -> list[str]:
        """Native label strings, in model output order."""

    @abstractmethod
    def to_canonical(self, label: str) -> str | None: ...

    @abstractmethod
    def predict_files(self, files: list[str], vocab: set[str]) -> dict[str, WindowScores]:
        """Score `files`, returning window-level scores for classes in `vocab` only."""

    def version_info(self) -> dict:
        """Anything that changes outputs. Goes into the cache key and the run record."""
        return {"adapter": type(self).__name__, **self.params}

    def cache_key(self) -> str:
        blob = json.dumps(self.version_info(), sort_keys=True, default=str)
        return f"{self.name}-{hashlib.sha1(blob.encode()).hexdigest()[:10]}"

    def coverage(self, vocab: set[str]) -> set[str]:
        return {c for c in map(self.to_canonical, self.labels()) if c in vocab}

"""Core data contracts shared by every model adapter and eval set.

Everything downstream (alignment, metrics, reports) only ever sees these two
shapes, so a new model or eval set just has to produce them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# Columns every eval-set segment table must have.
SEGMENT_COLUMNS = ["file", "start", "end", "labels"]


@dataclass
class EvalSet:
    """A set of labelled time segments over audio files.

    segments: DataFrame with columns
        file   (str)        absolute path to the audio file
        start  (float, s)   segment start within the file
        end    (float, s)   segment end within the file
        labels (list[str])  canonical species keys present (may be empty)
    classes: the canonical label vocabulary for this set. Metrics are computed
        over these classes; a class with zero positives is skipped by AP-style
        metrics but still counts towards coverage.
    """

    name: str
    segments: pd.DataFrame
    classes: list[str]
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        missing = set(SEGMENT_COLUMNS) - set(self.segments.columns)
        if missing:
            raise ValueError(f"EvalSet {self.name!r} segments missing columns: {missing}")
        self.segments = self.segments.reset_index(drop=True)
        unknown = {l for ls in self.segments["labels"] for l in ls} - set(self.classes)
        if unknown:
            raise ValueError(
                f"EvalSet {self.name!r} has labels outside its class list: {sorted(unknown)[:10]}"
            )

    @property
    def files(self) -> list[str]:
        return sorted(self.segments["file"].unique())

    def target_matrix(self) -> np.ndarray:
        """Binary [n_segments, n_classes] ground-truth matrix."""
        idx = {c: i for i, c in enumerate(self.classes)}
        y = np.zeros((len(self.segments), len(self.classes)), dtype=np.uint8)
        for r, labels in enumerate(self.segments["labels"]):
            for l in labels:
                y[r, idx[l]] = 1
        return y


@dataclass
class WindowScores:
    """Raw window-level model output for one audio file.

    starts/ends: [n_windows] seconds
    scores:      [n_windows, n_classes] float32, columns ordered as `classes`
    classes:     canonical species keys (only those the model can score)
    """

    file: str
    starts: np.ndarray
    ends: np.ndarray
    scores: np.ndarray
    classes: list[str]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            audio_file=np.array(self.file),
            starts=self.starts,
            ends=self.ends,
            scores=self.scores.astype(np.float32),
            classes=np.array(self.classes, dtype=object),
        )

    @classmethod
    def load(cls, path: Path) -> "WindowScores":
        d = np.load(path, allow_pickle=True)
        return cls(
            file=str(d["audio_file"]),
            starts=d["starts"],
            ends=d["ends"],
            scores=d["scores"],
            classes=list(d["classes"]),
        )

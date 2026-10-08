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
      and optionally
        recording (str)     source recording, when files are pieces of one
                            (BirdSet test_5s clips); used for bootstrap groups
        site (str)          recording site; bootstrap groups take precedence
        unreviewed (list[str])  species nobody checked in this segment. They
                            are unknown (-1), not absent, unless also in
                            `labels` (a positive annotation wins).
    classes: the canonical label vocabulary for this set. Metrics are computed
        over these classes; a class with zero positives is skipped by AP-style
        metrics but still counts towards coverage.
    events: optional call-level labels (file, start, end, label), in the same
        files and time base as `segments`. When present, recording-level
        evidence is checked against the calls themselves rather than the
        5 s segment labels.
    """

    name: str
    segments: pd.DataFrame
    classes: list[str]
    meta: dict = field(default_factory=dict)
    events: pd.DataFrame | None = None

    def __post_init__(self) -> None:
        missing = set(SEGMENT_COLUMNS) - set(self.segments.columns)
        if missing:
            raise ValueError(f"EvalSet {self.name!r} segments missing columns: {missing}")
        self.segments = self.segments.reset_index(drop=True)
        for col in ("labels", "unreviewed"):
            if col not in self.segments:
                continue
            unknown = {l for ls in self.segments[col] for l in ls} - set(self.classes)
            if unknown:
                raise ValueError(
                    f"EvalSet {self.name!r} has {col} outside its class list: {sorted(unknown)[:10]}"
                )
        if self.events is not None:
            missing = {"file", "start", "end", "label"} - set(self.events.columns)
            if missing:
                raise ValueError(f"EvalSet {self.name!r} events missing columns: {missing}")
            unknown = set(self.events["label"]) - set(self.classes)
            if unknown:
                raise ValueError(f"EvalSet {self.name!r} has events outside its class list: {sorted(unknown)[:10]}")
            self.events = self.events.reset_index(drop=True)

    def time_reviewed(self) -> np.ndarray:
        """Per segment: False if the segment is unreviewed for every class
        (unreviewed *time*, e.g. soundscape audio no labelled clip covers)."""
        if "unreviewed" not in self.segments:
            return np.ones(len(self.segments), bool)
        all_classes = set(self.classes)
        return np.array([not all_classes <= set(u) for u in self.segments["unreviewed"]])

    def groups(self) -> np.ndarray:
        """Bootstrap group per segment: site, else source recording, else file.

        Resampling whole groups keeps correlated segments together (e.g. the
        5 s clips of one BirdSet soundscape, or the chunks of one file).
        """
        for col in ("site", "recording"):
            if col in self.segments:
                return self.segments[col].to_numpy()
        return self.segments["file"].to_numpy()

    @property
    def files(self) -> list[str]:
        return sorted(self.segments["file"].unique())

    def target_matrix(self) -> np.ndarray:
        """[n_segments, n_classes] int8 ground truth: 1 present, 0 absent, -1 not reviewed."""
        idx = {c: i for i, c in enumerate(self.classes)}
        y = np.zeros((len(self.segments), len(self.classes)), dtype=np.int8)
        if "unreviewed" in self.segments:
            for r, unrev in enumerate(self.segments["unreviewed"]):
                y[r, [idx[l] for l in unrev]] = -1
        for r, labels in enumerate(self.segments["labels"]):
            y[r, [idx[l] for l in labels]] = 1
        return y

    def target_matrix_binary(self) -> np.ndarray:
        """The old binary form (unknown -> 0). Metrics use target_matrix()."""
        return (self.target_matrix() == 1).astype(np.uint8)


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

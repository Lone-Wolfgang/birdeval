"""BirdNET 2.4, BirdNET 3.0 and Perch V2 through the official `birdnet` package.

Written against birdnet 1.1.1, which loads all three:
    birdnet.load("acoustic", "2.4", "tf" | "pb")      48 kHz, 3 s windows
    birdnet.load("acoustic", "3.0", "onnx" | "pt" | ...)  32 kHz, 3 s default (variable)
    birdnet.load_perch_v2(device)                     32 kHz, 5 s windows, needs TensorFlow

For evaluation we need the full score vector, not detections, so every call
uses default_confidence_threshold=None and top_k = size of the species list.
Output is restricted to the eval set's vocabulary via custom_species_list to
keep memory small; sigmoid scores are per-class, so restricting does not change
any score.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from ..data import WindowScores
from ..taxonomy import Taxonomy
from .base import ModelAdapter

log = logging.getLogger(__name__)


class BirdNETLibModel(ModelAdapter):
    def __init__(
        self,
        name: str,
        taxonomy: Taxonomy,
        version: str,  # "2.4" | "3.0" | "perch_v2"
        backend: str | None = None,
        device: str = "CPU",
        overlap_s: float = 0.0,
        segment_size_s: float | None = None,  # 3.0 only
        batch_size: int = 16,
        n_workers: int | None = None,
        files_per_call: int = 50,
    ) -> None:
        super().__init__(
            name,
            taxonomy,
            version=version,
            backend=backend,
            overlap_s=overlap_s,
            segment_size_s=segment_size_s,
        )
        self.version = version
        self.backend = backend
        self.device = device
        self.overlap_s = overlap_s
        self.segment_size_s = segment_size_s
        self.batch_size = batch_size
        self.n_workers = n_workers
        self.files_per_call = files_per_call
        self._model = None

    # -- loading -------------------------------------------------------------
    @property
    def model(self):
        if self._model is None:
            import birdnet

            if self.version == "perch_v2":
                self._model = birdnet.load_perch_v2(self.device)
            else:
                self._model = birdnet.load("acoustic", self.version, self.backend or "tf")
        return self._model

    def version_info(self) -> dict:
        info = super().version_info()
        from importlib.metadata import PackageNotFoundError, version

        try:
            info["birdnet_pkg"] = version("birdnet")  # birdnet has no __version__
        except PackageNotFoundError:
            pass
        return info

    def labels(self) -> list[str]:
        return list(self.model.species_list)

    def to_canonical(self, label: str) -> str | None:
        # Perch V2 labels are bare scientific names; BirdNET's are "Sci_Common".
        if self.version == "perch_v2":
            return self.taxonomy.from_sci(label)
        return self.taxonomy.from_birdnet(label)

    # -- inference -------------------------------------------------------------
    def predict_files(self, files: list[str], vocab: set[str]) -> dict[str, WindowScores]:
        native = [l for l in self.labels() if self.to_canonical(l) in vocab]
        if not native:
            log.warning("%s covers none of the eval classes", self.name)
            return {f: _empty(f) for f in files}
        canon = [self.to_canonical(l) for l in native]

        kwargs = dict(
            # birdnet masks non-listed species to -inf *before* top-k, so
            # top_k=len(native) returns exactly the eval classes without
            # allocating the full 11-15k column tensor (verified in 1.1.1).
            top_k=len(native),
            default_confidence_threshold=None,
            custom_species_list=native,
            overlap_duration_s=self.overlap_s,
            batch_size=self.batch_size,
            n_workers=self.n_workers,
            device=self.device,
            apply_sigmoid=True,
            # 1.0 = plain sigmoid (zero bias). BirdNET defaults to this, but
            # Perch defaults to None, which birdnet rejects with apply_sigmoid.
            sigmoid_sensitivity=1.0,
        )
        if self.version == "3.0" and self.segment_size_s:
            kwargs["segment_size_s"] = self.segment_size_s

        out: dict[str, WindowScores] = {}
        for i in range(0, len(files), self.files_per_call):
            chunk = files[i : i + self.files_per_call]
            res = self.model.predict(chunk, **kwargs)
            out.update(_densify(res, native, canon))
        missing = set(files) - set(out)
        for f in missing:  # unreadable / too-short files
            log.warning("%s produced no output for %s", self.name, f)
            out[f] = _empty(f, canon)
        return out


def _empty(file: str, classes: list[str] | None = None) -> WindowScores:
    classes = classes or []
    return WindowScores(file, np.zeros(0), np.zeros(0), np.zeros((0, len(classes)), np.float32), classes)


def _densify(res, native: list[str], canon: list[str]) -> dict[str, WindowScores]:
    """Turn birdnet's sorted top-k tensor into dense [n_windows, n_classes] per file."""
    col = {l: j for j, l in enumerate(native)}
    names = np.asarray(res.species_list)
    ids = res.species_ids  # [n_files, n_seg, k]
    probs = res.species_probs
    masked = res.species_masked
    hop = float(res.hop_duration_s)
    seg_len = float(res.segment_duration_s)
    # map model species index -> our column (-1 if outside the restricted list)
    lut = np.array([col.get(str(n), -1) for n in names])

    out = {}
    for fi, path in enumerate(res.inputs):
        valid_seg = ~masked[fi].all(axis=1)
        n = int(valid_seg.sum())
        dense = np.zeros((n, len(native)), np.float32)
        for si in np.flatnonzero(valid_seg):
            keep = ~masked[fi, si]
            cols = lut[ids[fi, si][keep]]
            ok = cols >= 0
            dense[si, cols[ok]] = probs[fi, si][keep][ok]
        starts = np.flatnonzero(valid_seg) * hop
        dur = float(res.input_durations[fi])
        ends = np.minimum(starts + seg_len, dur)
        key = str(Path(str(path)).resolve())
        out[key] = WindowScores(key, starts, ends, dense, list(canon))
    return out

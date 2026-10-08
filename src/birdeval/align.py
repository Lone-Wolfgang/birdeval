"""Put window-level model scores onto the eval set's segment grid.

Models emit scores on their own windows (BirdNET 2.4: 3 s, Perch V2: 5 s, with
whatever hop you configure). Eval sets label fixed segments (BirdSet test_5s:
5 s). Each segment's score for a class is the MAX over model windows that
overlap it by at least `min_overlap_s`. The same rule is applied to every
model, so differences in results come from the models, not the windowing.

Classes in the eval set that a model cannot score get `missing_value`
(default 0.0) and are listed in the coverage report.
"""

from __future__ import annotations

import numpy as np

from .data import EvalSet, WindowScores


def align(
    evalset: EvalSet,
    windows: dict[str, WindowScores],
    min_overlap_s: float = 1.0,
    missing_value: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (scores [n_segments, n_classes], covered_mask [n_classes])."""
    classes = evalset.classes
    cidx = {c: i for i, c in enumerate(classes)}
    out = np.full((len(evalset.segments), len(classes)), missing_value, dtype=np.float32)
    covered = np.zeros(len(classes), dtype=bool)

    for file, seg in evalset.segments.groupby("file", sort=False):
        ws = windows.get(file)
        if ws is None:
            raise KeyError(f"No model output for {file}")
        # model columns that exist in the eval vocabulary
        model_cols = [j for j, c in enumerate(ws.classes) if c in cidx]
        eval_cols = [cidx[ws.classes[j]] for j in model_cols]
        covered[eval_cols] = True
        if not model_cols or len(ws.starts) == 0:
            continue
        sub = ws.scores[:, model_cols]
        for row, s, e in zip(seg.index, seg["start"].to_numpy(), seg["end"].to_numpy()):
            overlap = np.minimum(ws.ends, e) - np.maximum(ws.starts, s)
            # always accept at least the best-overlapping window, even if short
            hit = overlap >= min(min_overlap_s, max(overlap.max(), 0.0))
            hit &= overlap > 0
            if hit.any():
                out[row, eval_cols] = sub[hit].max(axis=0)
    return out, covered

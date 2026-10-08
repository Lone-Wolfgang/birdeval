"""Put window-level model scores onto the eval set's segment grid.

Models emit scores on their own windows (BirdNET 2.4: 3 s, Perch V2: 5 s, with
whatever hop you configure). Eval sets label fixed segments (BirdSet test_5s:
5 s). Each segment's score for a class is the MAX over the model windows that
count towards it.

Which windows count is decided by `window_hits`, shared with recording-level
unit membership:

    alignment:
      rule: half_window    # default: more than half of the window lies inside
                           # (strict >), measured on each window's own length
      rule: min_overlap    # the old rule: overlap >= min_overlap_s

half_window needs no per-model setting and never lets a window count towards
two neighbouring segments (no window has more than half of itself in two
places). With the old global min_overlap_s = 1.0, Perch's 5 s windows at a
2.5 s hop scored calls in the neighbouring segment as well; any single global
threshold was either too loose for Perch or too strict for BirdNET's 3 s
windows. If no window qualifies (e.g. a segment shorter than half a window),
the best-overlapping window counts.

Classes in the eval set that a model cannot score, and segments no window
reaches, get `missing_value` (the scorer's floor) and are listed in the
coverage report.
"""

from __future__ import annotations

import numpy as np

from .data import EvalSet, WindowScores

RULES = ("half_window", "min_overlap")


def window_hits(
    starts: np.ndarray,
    ends: np.ndarray,
    spans: np.ndarray,
    rule: str = "half_window",
    min_overlap_s: float = 1.0,
) -> np.ndarray:
    """Bool mask: which windows count towards the time covered by `spans` ([n, 2])."""
    spans = np.asarray(spans, float).reshape(-1, 2)
    overlap = np.zeros(len(starts))
    for s, e in spans:
        overlap += np.clip(np.minimum(ends, e) - np.maximum(starts, s), 0.0, None)
    if rule == "half_window":
        hit = overlap > (ends - starts) / 2
    elif rule == "min_overlap":
        hit = (overlap >= min_overlap_s) & (overlap > 0)
    else:
        raise ValueError(f"unknown alignment rule {rule!r}; expected one of {RULES}")
    if not hit.any() and len(overlap) and overlap.max() > 0:
        hit = overlap == overlap.max()  # fall back to the best-overlapping window
    return hit


def align(
    evalset: EvalSet,
    windows: dict[str, WindowScores],
    rule: str = "half_window",
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
            hit = window_hits(ws.starts, ws.ends, [(s, e)], rule, min_overlap_s)
            if hit.any():
                out[row, eval_cols] = sub[hit].max(axis=0)
    return out, covered

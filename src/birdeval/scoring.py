"""Scorer stage: turn cached raw logits into the scores a run evaluates.

Sits between the prediction cache and alignment / aggregation. It is a pure
function on WindowScores, so changing the method never re-runs inference.

    scoring:
      method: logit      # logit | sigmoid

- logit:   identity (the default; every ranking metric is computed on logits).
- sigmoid: 1 / (1 + exp(-x)), numerically stable and unclipped. Computed in
           float64 because float32 sigmoid saturates to exactly 1.0 above
           x ~ 17, which would re-create the ties raw logits avoid.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .data import WindowScores

METHODS = ("logit", "sigmoid")

# A score below anything the method can produce, for classes a model can't
# score and segments no window covers. Ranks last without being infinite
# (sklearn rejects inf).
FLOOR = {"logit": -1e6, "sigmoid": 0.0}


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    e = np.exp(x[~pos])
    out[~pos] = e / (1.0 + e)
    return out


def apply(ws: WindowScores, method: str = "logit") -> WindowScores:
    if method == "logit":
        return ws
    if method == "sigmoid":
        return replace(ws, scores=sigmoid(ws.scores))
    raise ValueError(f"unknown scoring method {method!r}; expected one of {METHODS}")


def floor(method: str = "logit") -> float:
    return FLOOR[method]

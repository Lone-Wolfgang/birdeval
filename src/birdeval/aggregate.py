"""Recording-level presence: pool window scores into (unit, species) claims.

A *unit* is what a presence claim is about: a whole file (unit_s = None), or
fixed-length chunks of it (unit_s = 60, 300, ...) to match a deployment's
recording length.

Ground truth per (unit, species), from the segment labels:
    1   any segment in the unit is present
    0   every segment in the unit is absent
   -1   no present segment and at least one unreviewed one (excluded from metrics)
A segment belongs to the unit containing its midpoint.

Features come from the scored model windows, not the aligned segment matrix,
so the evidence times are the model's real windows. A window belongs to a unit
if it overlaps it by >= min_overlap_s (or covers all of a shorter unit).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .data import EvalSet, WindowScores

FEATURES = ("max_score", "top3_mean", "n_above")


@dataclass
class UnitScores:
    """Per-(unit, class) features over an eval set's classes. Arrays are [n_units, n_classes]."""

    units: pd.DataFrame  # unit, file, start, end, duration_s, n_windows, group
    classes: list[str]
    covered: np.ndarray  # [n_classes] bool: the model can score this class
    max_score: np.ndarray
    top3_mean: np.ndarray
    n_above: np.ndarray
    evidence_start: np.ndarray
    evidence_end: np.ndarray


def make_units(ev: EvalSet, unit_s: float | None = None) -> tuple[pd.DataFrame, np.ndarray]:
    """Return (units table, unit index of every segment)."""
    seg = ev.segments
    group_col = "site" if "site" in seg else "file"
    mid = (seg["start"].to_numpy() + seg["end"].to_numpy()) / 2
    rows, seg_unit = [], np.empty(len(seg), dtype=int)
    for file, g in seg.groupby("file", sort=True):
        lo, hi = float(g["start"].min()), float(g["end"].max())
        group = g[group_col].iloc[0]
        if unit_s is None:
            seg_unit[g.index] = len(rows)
            rows.append((file, lo, hi, group))
            continue
        k = np.floor((mid[g.index] - lo) / unit_s).astype(int)
        for j in np.unique(k):  # chunks with no labelled segment are skipped
            seg_unit[g.index[k == j]] = len(rows)
            rows.append((file, lo + j * unit_s, min(lo + (j + 1) * unit_s, hi), group))
    units = pd.DataFrame(rows, columns=["file", "start", "end", "group"])
    units.insert(0, "unit", [f"{Path(f).name}@{s:g}" for f, s in zip(units.file, units.start)])
    units["duration_s"] = units["end"] - units["start"]
    return units, seg_unit


def unit_truth(y: np.ndarray, seg_unit: np.ndarray, n_units: int) -> np.ndarray:
    """Collapse three-state segment truth [n_seg, C] to unit truth [n_units, C]."""
    t = np.zeros((n_units, y.shape[1]), np.int8)
    any_pos = np.zeros_like(t, dtype=bool)
    any_unk = np.zeros_like(t, dtype=bool)
    np.logical_or.at(any_pos, seg_unit, y == 1)
    np.logical_or.at(any_unk, seg_unit, y == -1)
    t[any_unk] = -1
    t[any_pos] = 1
    return t


def aggregate(
    ev: EvalSet,
    windows: dict[str, WindowScores],
    units: pd.DataFrame,
    tau: float = 0.0,
    min_overlap_s: float = 1.0,
    floor: float = 0.0,
) -> UnitScores:
    classes = ev.classes
    cidx = {c: i for i, c in enumerate(classes)}
    U, C = len(units), len(classes)
    mx = np.full((U, C), floor, np.float64)
    t3 = np.full((U, C), floor, np.float64)
    n_above = np.zeros((U, C), np.int32)
    ev_s = np.full((U, C), np.nan)
    ev_e = np.full((U, C), np.nan)
    n_win = np.zeros(U, np.int32)
    covered = np.zeros(C, bool)

    for u, (file, s, e) in enumerate(zip(units.file, units.start, units.end)):
        ws = windows[file]
        mcols = [j for j, c in enumerate(ws.classes) if c in cidx]
        ecols = [cidx[ws.classes[j]] for j in mcols]
        covered[ecols] = True
        if not mcols or len(ws.starts) == 0:
            continue
        overlap = np.minimum(ws.ends, e) - np.maximum(ws.starts, s)
        inside = np.flatnonzero((overlap > 0) & (overlap >= min(min_overlap_s, e - s)))
        n_win[u] = len(inside)
        if not len(inside):
            continue
        W = np.asarray(ws.scores[np.ix_(inside, mcols)], np.float64)
        best = W.argmax(axis=0)
        mx[u, ecols] = W[best, np.arange(len(mcols))]
        k = min(3, len(inside))
        t3[u, ecols] = np.partition(W, len(inside) - k, axis=0)[-k:].mean(axis=0)
        n_above[u, ecols] = (W >= tau).sum(axis=0)
        ev_s[u, ecols] = ws.starts[inside][best]
        ev_e[u, ecols] = ws.ends[inside][best]

    units = units.assign(n_windows=n_win)
    return UnitScores(units, list(classes), covered, mx, t3, n_above, ev_s, ev_e)


def evidence_truth(ev: EvalSet, y: np.ndarray, us: UnitScores, min_overlap_s: float = 1.0) -> np.ndarray:
    """[n_units, n_classes] int8: does the evidence window contain the species?

    1 if it overlaps a segment labelled present for that species by
    >= min_overlap_s (or all of a shorter window); else -1 if any overlapping
    segment is unreviewed for it, or it overlaps no labelled time; else 0.
    """
    E = np.full(us.max_score.shape, -1, np.int8)
    seg = ev.segments
    by_file = {f: g.index.to_numpy() for f, g in seg.groupby("file", sort=False)}
    s0, s1 = seg["start"].to_numpy(), seg["end"].to_numpy()
    for u, file in enumerate(us.units.file):
        rows = by_file[file]
        es, ee = us.evidence_start[u], us.evidence_end[u]  # [C]
        ok = np.isfinite(es)
        if not ok.any():
            continue
        c = np.flatnonzero(ok)
        ov = np.minimum(ee[c, None], s1[rows]) - np.maximum(es[c, None], s0[rows])  # [c, n_seg]
        hit = (ov > 0) & (ov >= np.minimum(min_overlap_s, ee[c] - es[c])[:, None])
        lab = y[rows][:, c].T  # [c, n_seg]
        pos = (hit & (lab == 1)).any(axis=1)
        unk = (hit & (lab == -1)).any(axis=1) | ~hit.any(axis=1)
        E[u, c] = np.where(pos, 1, np.where(unk, -1, 0))
    return E


def presence_table(
    us: UnitScores, truth: np.ndarray, conf: np.ndarray | None = None, evidence: np.ndarray | None = None
) -> pd.DataFrame:
    """Long table of (unit, species) rows for the classes the model covers."""
    conf = us.max_score if conf is None else conf
    cols = np.flatnonzero(us.covered)
    U = len(us.units)
    rep = lambda a: np.repeat(np.asarray(a), len(cols))  # noqa: E731
    return pd.DataFrame(
        {
            "unit": rep(us.units.unit),
            "file": rep(us.units.file),
            "unit_start": rep(us.units.start),
            "unit_end": rep(us.units.end),
            "species": np.tile(np.asarray(us.classes, dtype=object)[cols], U),
            "presence_conf": conf[:, cols].ravel(),
            "max_score": us.max_score[:, cols].ravel(),
            "top3_mean": us.top3_mean[:, cols].ravel(),
            "n_above": us.n_above[:, cols].ravel(),
            "n_windows": rep(us.units.n_windows),
            "evidence_start": us.evidence_start[:, cols].ravel(),
            "evidence_end": us.evidence_end[:, cols].ravel(),
            "truth": truth[:, cols].ravel(),
            **({"evidence_truth": evidence[:, cols].ravel()} if evidence is not None else {}),
        }
    )


def detections_table(pres: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """The deployment's detection table (one row per claim) plus truth columns."""
    claims = pres[pres["presence_conf"] >= threshold] if np.isfinite(threshold) else pres.iloc[:0]
    return (
        claims.rename(columns={"evidence_start": "start", "evidence_end": "end", "presence_conf": "confidence",
                               "truth": "presence_correct", "evidence_truth": "evidence_correct"})
        [["file", "start", "end", "species", "confidence", "unit", "presence_correct", "evidence_correct"]]
        .sort_values(["file", "start", "species"])
    )


def fp_review_table(pres: pd.DataFrame, threshold: float, n: int = 200) -> pd.DataFrame:
    """The n highest-scoring absent (unit, species) pairs, for an expert to listen to.

    Not limited to claims above the threshold, so there is a list to review
    even when no threshold reaches the precision target.
    """
    fp = pres[pres["truth"] == 0].sort_values("presence_conf", ascending=False, kind="stable").head(n)
    fp = fp.assign(above_threshold=fp["presence_conf"] >= threshold)
    return fp[["file", "evidence_start", "evidence_end", "species", "presence_conf", "unit",
               "n_above", "n_windows", "above_threshold"]]

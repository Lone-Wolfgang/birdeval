"""Recording-level presence: pool window scores into (unit, species) claims.

A *unit* is what a presence claim is about: a whole file (unit_s = None), or
fixed-length chunks of it (unit_s = 60, 300, ...) to match a deployment's
recording length.

Units are trimmed to reviewed time: segments unreviewed for every species
(e.g. soundscape audio no labelled clip covers) are left out, so a short
unreviewed tail doesn't make every absent species unknown for the whole
recording. A unit is the list of reviewed spans it covers.

Ground truth per (unit, species), from the reviewed segments:
    1   any segment in the unit is present
    0   every segment in the unit is absent
   -1   no present segment and at least one unreviewed for this species
        (species-level review gaps; excluded from metrics)
A segment belongs to the unit containing its midpoint.

Features come from the scored model windows, not the aligned segment matrix,
so the evidence times are the model's real windows. A window belongs to a unit
under the same rule as segment alignment (align.window_hits; by default more
than half of the window lies inside the unit).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .align import window_hits
from .data import EvalSet, WindowScores
from .evalsets.strong_labels import event_hits

FEATURES = ("max_score", "top3_mean", "n_above")


@dataclass
class UnitScores:
    """Per-(unit, class) features over an eval set's classes. Arrays are [n_units, n_classes]."""

    units: pd.DataFrame  # unit, file, start, end, group, spans, duration_s, n_windows
    classes: list[str]
    covered: np.ndarray  # [n_classes] bool: the model can score this class
    max_score: np.ndarray
    top3_mean: np.ndarray
    n_above: np.ndarray
    evidence_start: np.ndarray
    evidence_end: np.ndarray


def make_units(ev: EvalSet, unit_s: float | None = None) -> tuple[pd.DataFrame, np.ndarray]:
    """Return (units table, unit index of every segment; -1 = unreviewed time).

    Each unit's bootstrap `group` is its first segment's EvalSet.groups()
    value, so chunks (unit_s) of one recording are always resampled together.
    `duration_s` is the reviewed time in the unit.
    """
    seg = ev.segments
    seg_group = ev.groups()
    reviewed = ev.time_reviewed()
    mid = (seg["start"].to_numpy() + seg["end"].to_numpy()) / 2
    rows, seg_unit = [], np.full(len(seg), -1, dtype=int)
    for file, g in seg[reviewed].groupby("file", sort=True):
        lo = float(g["start"].min())
        group = seg_group[g.index[0]]
        k = np.zeros(len(g), int) if unit_s is None else np.floor((mid[g.index] - lo) / unit_s).astype(int)
        for j in np.unique(k):  # chunks with no reviewed segment are skipped
            members = g[k == j]
            seg_unit[members.index] = len(rows)
            spans = _merge(members["start"].to_numpy(), members["end"].to_numpy())
            rows.append((file, spans[0][0], spans[-1][1], group, spans))
    units = pd.DataFrame(rows, columns=["file", "start", "end", "group", "spans"])
    units.insert(0, "unit", [f"{Path(f).name}@{s:g}" for f, s in zip(units.file, units.start)])
    units["duration_s"] = [sum(e - s for s, e in sp) for sp in units["spans"]]
    return units, seg_unit


def _merge(starts: np.ndarray, ends: np.ndarray) -> list[tuple[float, float]]:
    """Sorted, merged (start, end) spans."""
    out: list[list[float]] = []
    for s, e in sorted(zip(starts.tolist(), ends.tolist())):
        if out and s <= out[-1][1] + 1e-9:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(s, e) for s, e in out]


def unit_truth(y: np.ndarray, seg_unit: np.ndarray, n_units: int) -> np.ndarray:
    """Collapse three-state segment truth [n_seg, C] to unit truth [n_units, C].

    Segments with seg_unit == -1 (unreviewed time, outside every unit) are ignored.
    """
    t = np.zeros((n_units, y.shape[1]), np.int8)
    any_pos = np.zeros_like(t, dtype=bool)
    any_unk = np.zeros_like(t, dtype=bool)
    keep = seg_unit >= 0
    np.logical_or.at(any_pos, seg_unit[keep], y[keep] == 1)
    np.logical_or.at(any_unk, seg_unit[keep], y[keep] == -1)
    t[any_unk] = -1
    t[any_pos] = 1
    return t


def aggregate(
    ev: EvalSet,
    windows: dict[str, WindowScores],
    units: pd.DataFrame,
    tau: float = 0.0,
    rule: str = "half_window",
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

    for u, (file, spans) in enumerate(zip(units.file, units.spans)):
        ws = windows[file]
        mcols = [j for j, c in enumerate(ws.classes) if c in cidx]
        ecols = [cidx[ws.classes[j]] for j in mcols]
        covered[ecols] = True
        if not mcols or len(ws.starts) == 0:
            continue
        inside = np.flatnonzero(window_hits(ws.starts, ws.ends, spans, rule, min_overlap_s))
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


def evidence_truth(
    ev: EvalSet,
    y: np.ndarray,
    us: UnitScores,
    rule: str = "half_window",
    min_overlap_s: float = 1.0,
    min_event_overlap_s: float = 0.5,
) -> np.ndarray:
    """[n_units, n_classes] int8: does the evidence window contain the species?

    With call-level labels (ev.events): 1 if the window overlaps a call of
    that species by >= min_event_overlap_s (or contains a shorter call whole;
    the same rule that builds segment labels); else -1 if the window touches
    time unreviewed for that species, or no reviewed time at all; else 0.

    Without them, the 5 s segment labels: 1 if the window counts towards
    (align.window_hits) a segment labelled present; else -1 if any segment it
    counts towards is unreviewed for it, or it counts towards no labelled
    time; else 0.
    """
    if ev.events is not None:
        return _evidence_from_events(ev, y, us, min_event_overlap_s)
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
        ov = np.clip(np.minimum(ee[c, None], s1[rows]) - np.maximum(es[c, None], s0[rows]), 0, None)  # [c, n_seg]
        if rule == "half_window":
            hit = ov > ((ee[c] - es[c]) / 2)[:, None]
        else:
            hit = (ov >= min_overlap_s) & (ov > 0)
        # same fallback as window_hits: the best-overlapping segment
        best = (ov == ov.max(axis=1, keepdims=True)) & (ov > 0)
        hit = np.where(hit.any(axis=1, keepdims=True), hit, best)
        lab = y[rows][:, c].T  # [c, n_seg]
        pos = (hit & (lab == 1)).any(axis=1)
        unk = (hit & (lab == -1)).any(axis=1) | ~hit.any(axis=1)
        E[u, c] = np.where(pos, 1, np.where(unk, -1, 0))
    return E


def _evidence_from_events(ev: EvalSet, y: np.ndarray, us: UnitScores, min_event_overlap_s: float) -> np.ndarray:
    E = np.full(us.max_score.shape, -1, np.int8)
    seg = ev.segments
    by_file = {f: g.index.to_numpy() for f, g in seg.groupby("file", sort=False)}
    calls = {k: (g["start"].to_numpy(float), g["end"].to_numpy(float))
             for k, g in ev.events.groupby(["file", "label"], sort=False)}
    s0, s1 = seg["start"].to_numpy(), seg["end"].to_numpy()
    for u, file in enumerate(us.units.file):
        rows = by_file[file]
        for c in np.flatnonzero(np.isfinite(us.evidence_start[u])):
            es, ee = us.evidence_start[u, c], us.evidence_end[u, c]
            cs, ce = calls.get((file, us.classes[c]), (np.zeros(0), np.zeros(0)))
            if event_hits(cs, ce, es, ee, min_event_overlap_s).any():
                E[u, c] = 1
                continue
            touches = (np.minimum(ee, s1[rows]) - np.maximum(es, s0[rows])) > 0
            lab = y[rows, c]
            unknown = (touches & (lab == -1)).any() or not (touches & (lab != -1)).any()
            E[u, c] = -1 if unknown else 0
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

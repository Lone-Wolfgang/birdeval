"""Eval sets from your own annotations.

Two entry points:

load_strong_labels  - event annotations (file, start, end, label), e.g. Raven
                      selection tables merged into one CSV. Each file is cut
                      into fixed segments (default 5 s, same as BirdSet
                      test_5s) and a segment is positive for a species if an
                      event overlaps it by >= min_event_overlap_s.

load_segments_csv   - already-segmented labels: file, start, end, labels
                      (labels separated by ';', empty = negative).

Label strings can be scientific names (label_type: sci), eBird codes
(label_type: ebird) or BirdNET-style "Sci_Common" (label_type: birdnet).

Both take two options for data that wasn't exhaustively labelled. Anything
not reviewed is unknown (-1) rather than absent, and metrics skip it:

    reviewed_species: all     # or a list; other species are unknown everywhere
    reviewed_ranges: null     # CSV file,start,end; a segment not fully inside
                              # a reviewed range is unknown for every species

With reviewed_ranges, files listed there but with no annotations are included
as reviewed-and-empty, i.e. negatives.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from ..data import EvalSet
from ..taxonomy import Taxonomy

log = logging.getLogger(__name__)


def _resolve(audio_root: str | None, f: str) -> str:
    p = Path(f)
    if audio_root and not p.is_absolute():
        p = Path(audio_root) / p
    return str(p.resolve())


def _map_labels(labels: pd.Series, taxonomy: Taxonomy, label_type: str) -> tuple[pd.Series, list]:
    uniq = sorted({l for l in labels if isinstance(l, str) and l})
    mapped, unmapped = taxonomy.map_many(uniq, kind=label_type)
    lut = dict(zip(uniq, mapped))
    return labels.map(lambda l: lut.get(l) if isinstance(l, str) else None), unmapped


def _reviewed_species(reviewed_species, taxonomy: Taxonomy, label_type: str) -> set[str] | None:
    if reviewed_species in (None, "all"):
        return None
    mapped, unmapped = taxonomy.map_many(list(reviewed_species), kind=label_type)
    if unmapped:
        raise ValueError(f"reviewed_species not in taxonomy: {unmapped}")
    return set(mapped)


def _read_ranges(path: str | None, audio_root: str | None) -> dict[str, np.ndarray] | None:
    """file -> merged [n, 2] array of reviewed (start, end) intervals."""
    if not path:
        return None
    df = pd.read_csv(path)
    out = {}
    for f, g in df.groupby(df["file"].map(lambda f: _resolve(audio_root, f))):
        iv = sorted(zip(g["start"].astype(float), g["end"].astype(float)))
        merged = [list(iv[0])]
        for s, e in iv[1:]:
            if s <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e])
        out[f] = np.array(merged)
    return out


def _fully_reviewed(iv: np.ndarray | None, s: float, e: float) -> bool:
    if iv is None:
        return False
    covered = np.clip(np.minimum(iv[:, 1], e) - np.maximum(iv[:, 0], s), 0, None).sum()
    return covered >= (e - s) - 1e-6


def _mark_unreviewed(
    seg: pd.DataFrame,
    classes: list[str],
    reviewed: set[str] | None,
    ranges: dict[str, np.ndarray] | None,
) -> pd.DataFrame:
    """Add the `unreviewed` column. Without either option, leave seg untouched."""
    if reviewed is None and ranges is None:
        return seg
    not_reviewed = sorted(set(classes) - reviewed) if reviewed is not None else []
    seg = seg.copy()
    seg["unreviewed"] = [
        not_reviewed if ranges is None or _fully_reviewed(ranges.get(f), s, e) else list(classes)
        for f, s, e in zip(seg["file"], seg["start"], seg["end"])
    ]
    return seg


def load_strong_labels(
    name: str,
    taxonomy: Taxonomy,
    annotations: str,
    audio_root: str | None = None,
    label_type: str = "sci",
    columns: dict | None = None,
    segment_s: float = 5.0,
    min_event_overlap_s: float = 0.5,
    drop_partial_last: bool = True,
    max_files: int | None = None,
    reviewed_species: str | list[str] = "all",
    reviewed_ranges: str | None = None,
) -> EvalSet:
    cols = {"file": "file", "start": "start", "end": "end", "label": "label", **(columns or {})}
    ann = pd.read_csv(annotations)
    ann = ann.rename(columns={v: k for k, v in cols.items()})[list(cols)]
    ann["file"] = ann["file"].map(lambda f: _resolve(audio_root, f))
    ann["label"], unmapped = _map_labels(ann["label"], taxonomy, label_type)
    ann = ann.dropna(subset=["label"])

    ranges = _read_ranges(reviewed_ranges, audio_root)
    files = sorted(set(ann["file"]) | set(ranges or {}))
    if max_files:
        files = files[:max_files]

    rows = []
    for f in files:
        dur = sf.info(f).duration
        n = int(np.floor(dur / segment_s)) if drop_partial_last else int(np.ceil(dur / segment_s))
        ev = ann[ann["file"] == f]
        for k in range(max(n, 1)):
            s, e = k * segment_s, min((k + 1) * segment_s, dur)
            ov = np.minimum(ev["end"], e) - np.maximum(ev["start"], s)
            # an event shorter than the overlap threshold still counts if fully inside
            short_inside = (ev["end"] - ev["start"] < min_event_overlap_s) & (ov >= ev["end"] - ev["start"])
            hit = (ov >= min_event_overlap_s) | (short_inside & (ov > 0))
            rows.append((f, s, e, sorted(set(ev.loc[hit, "label"]))))

    seg = pd.DataFrame(rows, columns=["file", "start", "end", "labels"])
    reviewed = _reviewed_species(reviewed_species, taxonomy, label_type)
    classes = sorted(set(ann["label"]) | (reviewed or set()))
    seg = _mark_unreviewed(seg, classes, reviewed, ranges)
    return EvalSet(name, seg, classes, meta={"source": "strong_labels", "unmapped": unmapped})


def load_segments_csv(
    name: str,
    taxonomy: Taxonomy,
    path: str,
    audio_root: str | None = None,
    label_type: str = "sci",
    sep: str = ";",
    classes: list[str] | None = None,
    reviewed_species: str | list[str] = "all",
    reviewed_ranges: str | None = None,
) -> EvalSet:
    df = pd.read_csv(path, keep_default_na=False)
    df["file"] = df["file"].map(lambda f: _resolve(audio_root, f))
    raw = df["labels"].map(lambda s: [x.strip() for x in str(s).split(sep) if x.strip()])
    uniq = sorted({l for ls in raw for l in ls})
    mapped, unmapped = taxonomy.map_many(uniq, kind=label_type)
    lut = dict(zip(uniq, mapped))
    df["labels"] = raw.map(lambda ls: sorted({lut[l] for l in ls if lut[l]}))
    vocab = sorted({l for ls in df["labels"] for l in ls})
    if classes:
        vocab = sorted(set(vocab) | {taxonomy.from_sci(c) for c in classes})
    reviewed = _reviewed_species(reviewed_species, taxonomy, label_type)
    vocab = sorted(set(vocab) | (reviewed or set()))
    seg = _mark_unreviewed(
        df[["file", "start", "end", "labels"]], vocab, reviewed, _read_ranges(reviewed_ranges, audio_root)
    )
    return EvalSet(name, seg, vocab, meta={"unmapped": unmapped})

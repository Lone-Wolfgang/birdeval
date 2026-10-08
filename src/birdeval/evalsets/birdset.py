"""BirdSet test sets from Hugging Face (DBD-research-group/BirdSet).

Two modes, both on BirdSet's official 5 s labels:

load_birdset (loader: birdset)
    The `test_5s` split as published: every row is a separate, pre-cut 5 s
    audio file (e.g. Recording_1_Segment_03_000_005.ogg). Comparable to
    BirdSet's reported cmAP / AUROC / top-1. Each clip is its own file, so
    recording-level results here are per clip; a `recording` column
    (parsed from the filename) makes the bootstrap resample whole
    soundscapes rather than clips.

load_birdset_soundscapes (loader: birdset_soundscape)
    Models run on the full soundscapes from the `test` split, as in
    deployment. Each test_5s clip's labels are placed on its soundscape's
    timeline at the offset in its filename (`_SSS_EEE`), with the clip's true
    length from `end_time` (last clips can be partial). Time no clip covers
    becomes segments that are not reviewed for any species.

Both modes also carry BirdSet's call-level labels (the `test` split's
events) as EvalSet.events, shifted onto each clip in clip mode, so
recording-level evidence is checked against the calls themselves.

Both use the `<config>_scape` builder, which downloads only the test
material. The plain `<config>` builder also downloads the training audio
(~16 GB for POW alone).

BirdSet ships a loading script; `datasets` >= 4.0 dropped script support, so
this needs `datasets<4` and trust_remote_code=True.

Config names: HSN, NBP, NES, PER, POW, SNE, SSW, UHH.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from ..data import EvalSet
from ..taxonomy import Taxonomy

log = logging.getLogger(__name__)

_COLS = {
    "file": "filepath",
    "start": "start_time",
    "end": "end_time",
    "labels": "ebird_code_multilabel",
}

# test_5s clip name -> (soundscape name, offset). Holds for all 8 test sets.
_CLIP = re.compile(r"^(?P<scape>.+)_(?P<start>\d+)_(?P<end>\d+)(?P<ext>\.\w+)$")

SEGMENT_S = 5.0
_TOL_S = 0.01  # float slack when comparing clip ends to soundscape durations


def parse_clip_name(path: str) -> tuple[str, float] | None:
    """'.../Recording_1_Segment_03_005_010.ogg' -> ('Recording_1_Segment_03.ogg', 5.0)."""
    m = _CLIP.match(Path(path).name)
    return (m["scape"] + m["ext"], float(m["start"])) if m else None


def _load(config: str, split: str, cache_dir: str | None):
    from datasets import load_dataset

    ds = load_dataset(
        "DBD-research-group/BirdSet",
        f"{config}_scape",
        split=split,
        cache_dir=cache_dir,
        trust_remote_code=True,
    )
    if "audio" in ds.column_names:
        ds = ds.remove_columns("audio")  # we only need paths; skip decoding
    return ds


def _code_map(code_names: list[str], taxonomy: Taxonomy, config: str) -> tuple[dict[int, str | None], list[str]]:
    mapped, unmapped = taxonomy.map_many(list(code_names), kind="ebird")
    if unmapped:
        log.warning(
            "BirdSet %s: %d/%d eBird codes not in taxonomy CSV and will be dropped: %s",
            config, len(unmapped), len(code_names), unmapped[:10],
        )
    return dict(enumerate(mapped)), unmapped


def _clip_table(config: str, taxonomy: Taxonomy, cache_dir: str | None) -> tuple[pd.DataFrame, list[str], list[str]]:
    """test_5s rows: file, start, end, labels (canonical), plus classes and unmapped codes."""
    ds = _load(config, "test_5s", cache_dir)
    code_names = ds.features[_COLS["labels"]].feature.names
    df = ds.select_columns(list(_COLS.values())).to_pandas()
    df = df.rename(columns={v: k for k, v in _COLS.items()})
    df["file"] = df["file"].map(lambda p: str(Path(p).resolve()))
    id_to_canon, unmapped = _code_map(code_names, taxonomy, config)
    df["labels"] = df["labels"].map(lambda ids: sorted({id_to_canon[int(i)] for i in ids if id_to_canon[int(i)]}))
    classes = sorted({c for c in id_to_canon.values() if c})
    return df, classes, unmapped


def _test_table(config: str, taxonomy: Taxonomy, cache_dir: str | None) -> tuple[dict[str, str], pd.DataFrame]:
    """Soundscape paths by file name, and call-level events (scape, start, end, label)."""
    ds = _load(config, "test", cache_dir)
    names = ds.features["ebird_code"].names
    mapped, _ = taxonomy.map_many(list(names), kind="ebird")  # unmapped codes are reported by _clip_table
    df = ds.select_columns(["filepath", "start_time", "end_time", "ebird_code"]).to_pandas()
    paths = {Path(p).name: str(Path(p).resolve()) for p in df["filepath"].unique()}
    events = pd.DataFrame({
        "scape": df["filepath"].map(lambda p: Path(p).name),
        "start": df["start_time"].astype(float),
        "end": df["end_time"].astype(float),
        "label": df["ebird_code"].map(lambda i: mapped[int(i)] if pd.notna(i) else None),
    })
    return paths, events.dropna(subset=["label"]).reset_index(drop=True)


def events_on_clips(clips: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Shift soundscape events onto the test_5s clips they overlap (clip time base)."""
    by_scape = {k: g for k, g in events.groupby("scape", sort=False)}
    rows = []
    for f, s, e in zip(clips["file"], clips["start"], clips["end"]):
        scape, offset = parse_clip_name(f)
        g = by_scape.get(scape)
        if g is None:
            continue
        hit = (g["end"] > offset) & (g["start"] < offset + (e - s))
        for cs, ce, lab in zip(g["start"][hit], g["end"][hit], g["label"][hit]):
            rows.append((f, cs - offset, ce - offset, lab))
    return pd.DataFrame(rows, columns=["file", "start", "end", "label"])


def load_birdset(
    name: str,
    taxonomy: Taxonomy,
    config: str,
    split: str = "test_5s",
    cache_dir: str | None = None,
    max_files: int | None = None,
) -> EvalSet:
    if split != "test_5s":
        raise ValueError("loader `birdset` reads test_5s; use `birdset_soundscape` for full soundscapes")
    df, classes, unmapped = _clip_table(config, taxonomy, cache_dir)
    df["recording"] = df["file"].map(lambda f: (parse_clip_name(f) or (f,))[0])

    if max_files:
        keep = sorted(df["file"].unique())[:max_files]
        df = df[df["file"].isin(keep)]

    _, events = _test_table(config, taxonomy, cache_dir)
    events = events_on_clips(df, events[events["label"].isin(classes)])
    return EvalSet(
        name=name,
        segments=df[["file", "start", "end", "labels", "recording"]],
        classes=classes,
        meta={"source": "birdset", "mode": "clips", "config": config, "split": split, "unmapped_codes": unmapped},
        events=events,
    )


def clips_to_soundscape(clips: pd.DataFrame, scape_paths: dict[str, str]) -> pd.DataFrame:
    """Place clip rows (file, start=0, end=clip length, labels) on their soundscapes."""
    parsed = clips["file"].map(parse_clip_name)
    bad = parsed.isna()
    if bad.any():
        raise ValueError(f"{bad.sum()} test_5s clip names don't match {_CLIP.pattern}: {clips.file[bad].iloc[0]}")
    scape = parsed.map(lambda p: p[0])
    missing = sorted(set(scape) - set(scape_paths))
    if missing:
        raise ValueError(f"{len(missing)} soundscapes referenced by test_5s are not in the test split: {missing[:3]}")
    offset = parsed.map(lambda p: p[1])
    return pd.DataFrame({
        "file": scape.map(scape_paths).to_numpy(),
        "start": offset.to_numpy(),
        "end": (offset + clips["end"] - clips["start"]).to_numpy(),
        "labels": clips["labels"].to_numpy(),
    }).sort_values(["file", "start"], ignore_index=True)


def uncovered_segments(seg: pd.DataFrame, durations: dict[str, float], classes: list[str]) -> pd.DataFrame:
    """Segments (<= 5 s) for soundscape time no clip covers, unreviewed for every class."""
    rows = []
    for file, g in seg.groupby("file", sort=False):
        edges = [0.0, *np.ravel(np.column_stack([g["start"], g["end"]])), durations[file]]
        for lo, hi in zip(edges[0::2], edges[1::2]):  # gaps: before, between, after clips
            t = lo
            while hi - t > _TOL_S:
                rows.append((file, t, min(t + SEGMENT_S, hi), [], list(classes)))
                t += SEGMENT_S
    return pd.DataFrame(rows, columns=["file", "start", "end", "labels", "unreviewed"])


def load_birdset_soundscapes(
    name: str,
    taxonomy: Taxonomy,
    config: str,
    cache_dir: str | None = None,
    max_files: int | None = None,
) -> EvalSet:
    clips, classes, unmapped = _clip_table(config, taxonomy, cache_dir)
    scape_paths, events = _test_table(config, taxonomy, cache_dir)
    seg = clips_to_soundscape(clips, scape_paths)

    if max_files:
        seg = seg[seg["file"].isin(sorted(seg["file"].unique())[:max_files])]
    durations = {f: sf.info(f).duration for f in seg["file"].unique()}
    over = seg.groupby("file")["end"].max() - pd.Series(durations)
    if (over > _TOL_S).any():
        raise ValueError(f"test_5s clips run past their soundscape's end: {over[over > _TOL_S].head(3).to_dict()}")

    gaps = uncovered_segments(seg, durations, classes)
    seg = seg.assign(unreviewed=[[] for _ in range(len(seg))])
    if len(gaps):  # concat with an empty frame would turn start/end into object dtype
        seg = pd.concat([seg, gaps], ignore_index=True).sort_values(["file", "start"], ignore_index=True)
    uncovered_s = float((gaps["end"] - gaps["start"]).sum()) if len(gaps) else 0.0
    if uncovered_s:
        log.info("BirdSet %s soundscapes: %.1f s not covered by any test_5s clip (marked not reviewed) in %d/%d files",
                 config, uncovered_s, gaps["file"].nunique(), len(durations))

    events = events[events["label"].isin(classes)].assign(file=lambda d: d["scape"].map(scape_paths))
    events = events[events["file"].isin(durations)][["file", "start", "end", "label"]]
    return EvalSet(
        name=name,
        segments=seg,
        classes=classes,
        meta={"source": "birdset", "mode": "soundscapes", "config": config, "unmapped_codes": unmapped,
              "uncovered_s": uncovered_s, "files_with_uncovered_time": int(gaps["file"].nunique()) if len(gaps) else 0},
        events=events,
    )

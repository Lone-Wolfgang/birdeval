"""BirdSet test sets from Hugging Face (DBD-research-group/BirdSet).

Uses the `test_5s` split: each soundscape cut into 5 s segments with
multi-label eBird codes (empty list = no bird). This matches how BirdSet
reports cmAP / AUROC / top-1.

UNTESTED against the live dataset (Hugging Face wasn't reachable when this was
written). Things to check on first run, all in `_COLS` below:
  - column names (filepath / start_time / end_time / ebird_code_multilabel)
  - that `filepath` points at a local, readable audio file after download
  - BirdSet ships a loading script; recent `datasets` (>=4.0) dropped script
    support, so you may need `pip install "datasets<4"` and
    trust_remote_code=True.

Config names: HSN, NBP, NES, PER, POW, SNE, SSW, UHH.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from ..data import EvalSet
from ..taxonomy import Taxonomy

log = logging.getLogger(__name__)

_COLS = {
    "file": "filepath",
    "start": "start_time",
    "end": "end_time",
    "labels": "ebird_code_multilabel",
}


def load_birdset(
    name: str,
    taxonomy: Taxonomy,
    config: str,
    split: str = "test_5s",
    cache_dir: str | None = None,
    max_files: int | None = None,
) -> EvalSet:
    from datasets import load_dataset

    ds = load_dataset(
        "DBD-research-group/BirdSet",
        config,
        split=split,
        cache_dir=cache_dir,
        trust_remote_code=True,
    )
    if "audio" in ds.column_names:
        ds = ds.remove_columns("audio")  # we only need paths; skip decoding
    code_names = ds.features[_COLS["labels"]].feature.names

    df = ds.select_columns(list(_COLS.values())).to_pandas()
    df = df.rename(columns={v: k for k, v in _COLS.items()})
    df["file"] = df["file"].map(lambda p: str(Path(p).resolve()))

    mapped, unmapped = taxonomy.map_many(list(code_names), kind="ebird")
    id_to_canon = dict(enumerate(mapped))
    if unmapped:
        log.warning(
            "BirdSet %s: %d/%d eBird codes not in taxonomy CSV and will be dropped: %s",
            config, len(unmapped), len(code_names), unmapped[:10],
        )
    df["labels"] = df["labels"].map(
        lambda ids: sorted({id_to_canon[int(i)] for i in ids if id_to_canon[int(i)]})
    )
    classes = sorted({c for c in mapped if c})

    if max_files:
        keep = sorted(df["file"].unique())[:max_files]
        df = df[df["file"].isin(keep)]

    return EvalSet(
        name=name,
        segments=df[["file", "start", "end", "labels"]],
        classes=classes,
        meta={"source": "birdset", "config": config, "split": split, "unmapped_codes": unmapped},
    )

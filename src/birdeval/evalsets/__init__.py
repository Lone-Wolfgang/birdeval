from __future__ import annotations

import importlib

from ..data import EvalSet
from ..taxonomy import Taxonomy

LOADERS = {
    "birdset": "birdeval.evalsets.birdset:load_birdset",
    "birdset_soundscape": "birdeval.evalsets.birdset:load_birdset_soundscapes",
    "strong_labels": "birdeval.evalsets.strong_labels:load_strong_labels",
    "segments_csv": "birdeval.evalsets.strong_labels:load_segments_csv",
}


def build_evalset(name: str, cfg: dict, taxonomy: Taxonomy) -> EvalSet:
    """Build an eval set from a config entry: {loader: <key>, ...kwargs}."""
    cfg = dict(cfg)
    mod, fn = LOADERS[cfg.pop("loader")].split(":")
    return getattr(importlib.import_module(mod), fn)(name=name, taxonomy=taxonomy, **cfg)

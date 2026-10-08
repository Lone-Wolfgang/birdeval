from __future__ import annotations

from ..taxonomy import Taxonomy
from .base import ModelAdapter

ADAPTERS = {
    "birdnet": "birdeval.models.birdnet_lib:BirdNETLibModel",
    "fake": "birdeval.models.fake:FakeModel",
}


def build_model(name: str, cfg: dict, taxonomy: Taxonomy) -> ModelAdapter:
    """Build a model from a config entry: {adapter: <key>, ...kwargs}."""
    import importlib

    cfg = dict(cfg)
    target = ADAPTERS[cfg.pop("adapter")]
    mod, cls = target.split(":")
    return getattr(importlib.import_module(mod), cls)(name=name, taxonomy=taxonomy, **cfg)

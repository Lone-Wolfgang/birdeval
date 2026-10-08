"""Smoke-check that models load and produce sane window scores.

    birdeval check                                   # every model in the config
    birdeval check -m birdnet_2.4 --audio clip.wav   # plus top species on a real clip

For each model: build it from the config, load it (timing the first load,
which may download weights), count how many native labels map onto the
canonical label space, then run the same `predict_files` path the runner uses
on two generated clips (12 s, and 2 s which is shorter than any model window)
plus any --audio files. One model failing doesn't stop the others.
"""

from __future__ import annotations

import logging
import tempfile
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf

from .data import WindowScores
from .models import build_model
from .models.base import ModelAdapter
from .taxonomy import Taxonomy

log = logging.getLogger("birdeval")

SYNTH = {"synth_12s": 12.0, "synth_2s": 2.0}


@dataclass
class CheckResult:
    model: str
    ok: bool = False
    error: str = ""
    load_s: float = float("nan")
    n_labels: int = 0
    n_mapped: int = 0
    infer_s: float = float("nan")
    windows: dict[str, int] = field(default_factory=dict)  # clip name -> n windows
    window_s: float = float("nan")
    hop_s: float = float("nan")
    top: dict[str, list[tuple[str, float]]] = field(default_factory=dict)  # real clip -> top-k


def write_synthetic(dir: Path, sr: int = 48_000, seed: int = 0) -> dict[str, str]:
    """Noise plus a frequency-swept tone, so every model sees *something*."""
    rng = np.random.default_rng(seed)
    out = {}
    for name, dur in SYNTH.items():
        t = np.arange(int(sr * dur)) / sr
        x = 0.05 * rng.standard_normal(t.size) + 0.2 * np.sin(2 * np.pi * (2000 + 500 * t) * t)
        p = dir / f"{name}.wav"
        sf.write(p, x.astype(np.float32), sr)
        out[name] = str(p.resolve())
    return out


def _validate(ws: WindowScores, dur: float) -> None:
    s = ws.scores
    if s.ndim != 2 or s.shape != (len(ws.starts), len(ws.classes)):
        raise AssertionError(f"{Path(ws.file).name}: bad score shape {s.shape}")
    if len(ws.starts) == 0:
        raise AssertionError(f"{Path(ws.file).name}: no windows")
    if not np.isfinite(s).all() or s.min() < 0 or s.max() > 1:
        raise AssertionError(f"{Path(ws.file).name}: scores outside [0, 1]: {s.min():.3g}..{s.max():.3g}")
    if ws.ends.max() > dur + 1e-3:
        raise AssertionError(f"{Path(ws.file).name}: window ends {ws.ends.max():.2f}s past duration {dur:.2f}s")


def top_k(ws: WindowScores, k: int = 3) -> list[tuple[str, float]]:
    """Top classes by max score over windows."""
    m = ws.scores.max(axis=0)
    return [(ws.classes[j], float(m[j])) for j in np.argsort(-m)[:k]]


def check_model(model: ModelAdapter, synth: dict[str, str], audio: list[str]) -> CheckResult:
    r = CheckResult(model.name)
    try:
        t0 = time.time()
        labels = model.labels()  # triggers the (lazy) load
        r.load_s = time.time() - t0
        canon = {c for c in map(model.to_canonical, labels) if c}
        r.n_labels, r.n_mapped = len(labels), sum(model.to_canonical(l) is not None for l in labels)

        files = list(synth.values()) + audio
        t0 = time.time()
        out = model.predict_files(files, canon)
        r.infer_s = time.time() - t0

        names = {v: k for k, v in synth.items()} | {f: Path(f).name for f in audio}
        for f in files:
            if f not in out:
                raise AssertionError(f"no output for {names[f]}")
            ws = out[f]
            _validate(ws, sf.info(f).duration)
            r.windows[names[f]] = len(ws.starts)
        long = out[synth["synth_12s"]]
        r.window_s = float(long.ends[0] - long.starts[0])
        r.hop_s = float(long.starts[1] - long.starts[0]) if len(long.starts) > 1 else float("nan")
        for f in audio:
            r.top[names[f]] = top_k(out[f])
        r.ok = True
    except Exception as e:  # noqa: BLE001 - report every model, don't stop at the first
        r.error = f"{type(e).__name__}: {e}"
        log.error("%s failed:\n%s", model.name, traceback.format_exc())
    return r


def _taxonomy(cfg: dict) -> Taxonomy:
    t = dict(cfg.get("taxonomy", {}))
    if t.get("ebird_csv") and not Path(t["ebird_csv"]).exists():
        t["ebird_csv"] = None  # not needed to load and run a model
    return Taxonomy(**t)


def run_check(cfg: dict, model_names: list[str] | None = None, audio: list[str] | None = None) -> list[CheckResult]:
    tax = _taxonomy(cfg)
    audio = [str(Path(a).resolve()) for a in audio or []]
    results = []
    with tempfile.TemporaryDirectory() as d:
        synth = write_synthetic(Path(d))
        for name in model_names or list(cfg["models"]):
            log.info("Checking %s", name)
            try:
                model = build_model(name, cfg["models"][name], tax)
            except Exception as e:  # noqa: BLE001
                results.append(CheckResult(name, error=f"build: {type(e).__name__}: {e}"))
                continue
            results.append(check_model(model, synth, audio))
    return results


def format_report(results: list[CheckResult]) -> str:
    lines = [
        f"{'model':<16}{'status':<8}{'load s':>8}{'labels':>8}{'mapped':>8}{'infer s':>9}"
        f"{'win/hop s':>11}  windows"
    ]
    for r in results:
        win = " ".join(f"{k}={v}" for k, v in r.windows.items())
        lines.append(
            f"{r.model:<16}{'OK' if r.ok else 'FAIL':<8}{r.load_s:>8.1f}{r.n_labels:>8}{r.n_mapped:>8}"
            f"{r.infer_s:>9.1f}{f'{r.window_s:g}/{r.hop_s:g}':>11}  {win}"
        )
        if r.error:
            lines.append(f"    error: {r.error}")
        for clip, top in r.top.items():
            lines.append(f"    {clip}: " + ", ".join(f"{c} ({p:.2f})" for c, p in top))
    return "\n".join(lines)

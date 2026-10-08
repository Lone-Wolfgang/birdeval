"""Run any set of models against any set of eval sets.

    birdeval list -c configs/eval.yaml
    birdeval run  -c configs/eval.yaml -m birdnet_2.4 birdnet_3.0 perch_v2 -e birdset_HSN birdset_POW

Each (model, eval set) pair writes results/<run>/<evalset>/<model>/:
    summary.json   cmAP / AUROC / top-1, with file-level bootstrap CI for cmAP
    per_class.csv  AP / AUROC per species
and the run writes results/<run>/summary.csv with one row per pair.

Metrics are reported on three class sets so vocabulary differences don't
masquerade as accuracy differences:
    all      every class in the eval set (classes a model can't predict score 0)
    covered  classes this model can predict
    shared   classes every model in this run can predict  <- use this to compare
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from . import metrics, scoring
from .align import align
from .cache import PredictionCache
from .data import EvalSet
from .evalsets import build_evalset
from .models import build_model
from .models.base import ModelAdapter
from .taxonomy import Taxonomy

log = logging.getLogger("birdeval")


def load_config(path: str) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    base = Path(path).parent
    # resolve relative paths in the taxonomy section against the config file
    for k in ("ebird_csv", "synonyms_yaml"):
        v = cfg.get("taxonomy", {}).get(k)
        if v and not Path(v).is_absolute():
            cfg["taxonomy"][k] = str((base / v).resolve())
    return cfg


def alignment_config(cfg: dict, method: str) -> dict:
    """Alignment settings, with missing_value defaulting to the scorer's floor."""
    a = dict(cfg.get("alignment", {}))
    if a.get("missing_value") is None:
        a["missing_value"] = scoring.floor(method)
    elif method == "logit":
        log.warning(
            "alignment.missing_value=%s is a real score in logit space (0.0 = p 0.5); "
            "remove it to use the scorer floor",
            a["missing_value"],
        )
    return a


def get_window_scores(
    model: ModelAdapter, ev: EvalSet, cache_root: str, method: str = "logit"
) -> dict:
    cache = PredictionCache(cache_root, model.cache_key(), ev.name, ev.classes)
    todo = cache.missing(ev.files)
    if todo:
        log.info("%s on %s: running inference on %d/%d files", model.name, ev.name, len(todo), len(ev.files))
        t0 = time.time()
        for ws in model.predict_files(todo, set(ev.classes)).values():
            cache.put(ws)
        log.info("  done in %.1fs", time.time() - t0)
    else:
        log.info("%s on %s: all %d files cached", model.name, ev.name, len(ev.files))
    return {f: scoring.apply(cache.get(f), method) for f in ev.files}


def evaluate(
    cfg: dict,
    model_names: list[str],
    evalset_names: list[str],
    run_name: str | None = None,
    models: dict[str, ModelAdapter] | None = None,
) -> pd.DataFrame:
    tax = Taxonomy(**cfg.get("taxonomy", {}))
    score_cfg = {"method": "logit", **cfg.get("scoring", {})}
    align_cfg = alignment_config(cfg, score_cfg["method"])
    out_root = Path(cfg.get("output_dir", "results")) / (run_name or time.strftime("%Y%m%d-%H%M%S"))
    cache_root = cfg.get("cache_dir", "cache")
    n_boot = cfg.get("bootstrap", 200)

    models = models or {m: build_model(m, cfg["models"][m], tax) for m in model_names}
    rows = []
    for e in evalset_names:
        ev = build_evalset(e, cfg["evalsets"][e], tax) if isinstance(e, str) else e
        y = ev.target_matrix()
        groups = ev.segments["file"].to_numpy()
        log.info("Eval set %s: %d segments, %d files, %d classes", ev.name, len(y), len(ev.files), len(ev.classes))

        aligned, covered = {}, {}
        for name, model in models.items():
            ws = get_window_scores(model, ev, cache_root, score_cfg["method"])
            aligned[name], covered[name] = align(ev, ws, **align_cfg)
        shared = np.logical_and.reduce(list(covered.values()))

        for name in models:
            d = out_root / ev.name / name
            d.mkdir(parents=True, exist_ok=True)
            result = {
                "model": name,
                "evalset": ev.name,
                "model_info": models[name].version_info(),
                "evalset_meta": {k: v for k, v in ev.meta.items() if not isinstance(v, list)},
                "scoring": score_cfg,
                "alignment": align_cfg,
                "coverage": {"covered": int(covered[name].sum()), "shared": int(shared.sum()), "total": len(ev.classes)},
            }
            for scope, mask in (("all", None), ("covered", covered[name]), ("shared", shared)):
                summ, per_class = metrics.compute(y, aligned[name], ev.classes, restrict=mask)
                if n_boot:
                    summ["cmAP_ci95"] = metrics.bootstrap_cmap(y, aligned[name], groups, mask, n=n_boot)
                result[scope] = summ
                per_class.to_csv(d / f"per_class_{scope}.csv", index=False)
                rows.append({"evalset": ev.name, "model": name, "classes": scope, **summ})
            missing = [c for c, ok in zip(ev.classes, covered[name]) if not ok]
            (d / "uncovered_classes.txt").write_text("\n".join(missing))
            (d / "summary.json").write_text(json.dumps(result, indent=2, default=str))

    table = pd.DataFrame(rows)
    out_root.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_root / "summary.csv", index=False)
    log.info("Results written to %s", out_root)
    return table


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser("birdeval")
    sub = p.add_subparsers(dest="cmd", required=True)
    pl = sub.add_parser("list")
    pl.add_argument("-c", "--config", default="configs/eval.yaml")
    pr = sub.add_parser("run")
    pr.add_argument("-c", "--config", default="configs/eval.yaml")
    pr.add_argument("-m", "--models", nargs="+", required=True)
    pr.add_argument("-e", "--evalsets", nargs="+", required=True)
    pr.add_argument("--run-name")
    pc = sub.add_parser("check", help="load each model and run inference on test clips")
    pc.add_argument("-c", "--config", default="configs/eval.yaml")
    pc.add_argument("-m", "--models", nargs="+", help="default: every model in the config")
    pc.add_argument("--audio", nargs="+", default=[], help="real clips to show top species for")
    a = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(a.config)
    if a.cmd == "list":
        print("models:  ", ", ".join(cfg["models"]))
        print("evalsets:", ", ".join(cfg["evalsets"]))
        return
    if a.cmd == "check":
        from .check import format_report, run_check

        results = run_check(cfg, a.models, a.audio)
        print(format_report(results))
        raise SystemExit(0 if all(r.ok for r in results) else 1)
    table = evaluate(cfg, a.models, a.evalsets, a.run_name)
    with pd.option_context("display.width", 140, "display.max_columns", 20):
        print(table[table["classes"] == "shared"].drop(columns="classes").round(3).to_string(index=False))


if __name__ == "__main__":
    main()

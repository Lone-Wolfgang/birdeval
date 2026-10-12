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

from . import metrics, recording_metrics, scoring
from .aggregate import (
    aggregate,
    detections_table,
    evidence_truth,
    fp_review_table,
    make_units,
    presence_table,
    unit_truth,
)
from .align import align
from .cache import PredictionCache
from .data import EvalSet
from .evalsets import build_evalset
from .models import build_model
from .models.base import ModelAdapter
from .progress import Progress
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
    a = {"rule": "half_window", "min_overlap_s": 1.0, **cfg.get("alignment", {})}
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
        done = len(ev.files) - len(todo)
        log.info("%s on %s: running inference on %d/%d files%s", model.name, ev.name, len(todo), len(ev.files),
                 f" (resuming: {done} already cached)" if done else "")
        t0 = time.time()
        progress = Progress(len(todo), f"{model.name} on {ev.name}")

        def on_result(ws):
            # each file is cached as soon as it's scored, so a crash keeps it
            cache.put(ws)
            progress.update()

        try:
            out = model.predict_files(todo, set(ev.classes), on_result=on_result)
            # adapters that don't stream are cached (and counted) once the call returns
            for f, ws in out.items():
                if not cache.has(f):
                    cache.put(ws)
                    progress.update()
        finally:
            progress.close()
        log.info("  done in %.1fs", time.time() - t0)
    else:
        log.info("%s on %s: all %d files cached", model.name, ev.name, len(ev.files))
    return {f: scoring.apply(cache.get(f), method) for f in ev.files}


REC_DEFAULTS = {
    "enabled": True,
    "unit_s": None,
    "duration_bins_s": [60, 300, 900, 3600],
    "tau": 0.0,
    "precision_targets": [0.8, 0.9, 0.95],
    "operating_target": 0.9,  # which target's threshold detections.csv uses
    "fp_review_n": 200,
    "evidence_min_event_overlap_s": 0.5,  # call-level evidence: same rule as label building
}


def recording_level(ev, y, units, T, windows, scopes, rec, align_cfg, method, n_boot, d, workers=1) -> dict:
    """Write recording_level.json, presence.csv, detections.csv and fp_review.csv
    for one model; return headline columns per scope for summary.csv.

    The operating threshold for detections.csv is the one meeting
    `operating_target` on the `all` scope.
    """
    if rec["operating_target"] not in rec["precision_targets"]:
        raise ValueError(f"recording_level.operating_target {rec['operating_target']} "
                         f"is not one of precision_targets {rec['precision_targets']}")
    rule = {"rule": align_cfg["rule"], "min_overlap_s": align_cfg["min_overlap_s"]}  # same rule as segments
    us = aggregate(ev, windows, units, tau=rec["tau"], floor=scoring.floor(method), **rule)
    E = evidence_truth(ev, y, us, min_event_overlap_s=rec["evidence_min_event_overlap_s"], **rule)
    seg_s = (ev.segments["end"] - ev.segments["start"]).to_numpy()
    out = {
        "units": {"unit_s": rec["unit_s"], "n_units": len(units),
                  "reviewed_s": float(units["duration_s"].sum()),
                  "unreviewed_time_trimmed_s": float(seg_s[~ev.time_reviewed()].sum())},
        "config": rec,
    }
    headline = {}
    for scope, mask in scopes:
        r = recording_metrics.compute(
            T, us.max_score, units["duration_s"].to_numpy(), units["group"].to_numpy(), restrict=mask,
            targets=rec["precision_targets"], duration_bins_s=rec["duration_bins_s"], n_boot=n_boot, E=E,
            workers=workers,
        )
        out[scope] = r
        thresholds = {t: tr["threshold"] for t, tr in r["targets"].items()}
        cols = np.arange(len(ev.classes)) if mask is None else np.flatnonzero(mask)
        recording_metrics.per_species(T, us.max_score, ev.classes, cols, thresholds).to_csv(
            d / f"per_species_recording_{scope}.csv", index=False
        )
        h = {"rec_cmAP": r["cmAP"]}
        for t, tr in r["targets"].items():
            p = f"p{round(float(t) * 100)}"
            h |= {f"rec_recall_at_{p}": tr["recall"], f"rec_threshold_at_{p}": tr["threshold"],
                  f"rec_list_precision_at_{p}": tr["list_precision"],
                  f"rec_evidence_precision_at_{p}": tr["evidence_precision"]}
        headline[scope] = h
    op = out["all"]["targets"][str(rec["operating_target"])]
    if not op["reached"]:
        log.warning("%s: %s; detections.csv is empty", d.name, op["note"])
    pres = presence_table(us, T, evidence=E)
    pres.drop(columns="evidence_truth").to_csv(d / "presence.csv", index=False)
    detections_table(pres, op["threshold"]).to_csv(d / "detections.csv", index=False)
    fp_review_table(pres, op["threshold"], rec["fp_review_n"]).to_csv(d / "fp_review.csv", index=False)
    (d / "recording_level.json").write_text(json.dumps(out, indent=2, default=str))
    return headline


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
    workers = cfg.get("workers") or metrics.default_workers()
    rec = {**REC_DEFAULTS, **(cfg.get("recording_level") or {})}

    models = models or {m: build_model(m, cfg["models"][m], tax) for m in model_names}
    rows, table = [], pd.DataFrame()
    for e in evalset_names:
        ev = build_evalset(e, cfg["evalsets"][e], tax) if isinstance(e, str) else e
        y = ev.target_matrix()
        groups = ev.groups()
        log.info("Eval set %s: %d segments, %d files, %d classes", ev.name, len(y), len(ev.files), len(ev.classes))

        aligned, covered, windows = {}, {}, {}
        for name, model in models.items():
            windows[name] = get_window_scores(model, ev, cache_root, score_cfg["method"])
            aligned[name], covered[name] = align(ev, windows[name], **align_cfg)
        shared = np.logical_and.reduce(list(covered.values()))
        if rec["enabled"]:
            units, seg_unit = make_units(ev, rec["unit_s"])
            T = unit_truth(y, seg_unit, len(units))

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
            scopes = (("all", None), ("covered", covered[name]), ("shared", shared))
            headline = {}
            if rec["enabled"]:
                headline = recording_level(
                    ev, y, units, T, windows[name], scopes, rec, align_cfg, score_cfg["method"], n_boot, d, workers
                )
            for scope, mask in scopes:
                summ, per_class = metrics.compute(y, aligned[name], ev.classes, restrict=mask)
                if n_boot:
                    summ["cmAP_ci95"] = metrics.bootstrap_cmap(y, aligned[name], groups, mask, n=n_boot, workers=workers)
                result[scope] = summ
                per_class.to_csv(d / f"per_class_{scope}.csv", index=False)
                rows.append({"evalset": ev.name, "model": name, "classes": scope, **summ, **headline.get(scope, {})})
            missing = [c for c, ok in zip(ev.classes, covered[name]) if not ok]
            (d / "uncovered_classes.txt").write_text("\n".join(missing))
            (d / "summary.json").write_text(json.dumps(result, indent=2, default=str))
        # rewritten after every eval set, so a stopped run keeps what finished
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
    from tqdm.contrib.logging import logging_redirect_tqdm

    with logging_redirect_tqdm():  # keep log lines from breaking the progress bars
        table = evaluate(cfg, a.models, a.evalsets, a.run_name)
    with pd.option_context("display.width", 140, "display.max_columns", 20):
        print(table[table["classes"] == "shared"].drop(columns="classes").round(3).to_string(index=False))


if __name__ == "__main__":
    main()

"""Bird activity and model performance by local hour of day, from cached predictions.

    python -I scripts/hourly.py [--config configs/eval.yaml] [--sites HSN NES PER POW SNE SSW UHH]
                                [--out results/report/hourly.csv] [--n-boot 200]

Never runs inference. For each BirdSet soundscape site with clock times (not
NBP), it rebuilds the evaluation's segment x species truth and scores from the
prediction cache, using the same config, alignment and scoring as `birdeval run`.
A site is skipped with a message unless every model has every file cached,
in a cache that reproduces the per-class AP of the run the report uses
(--runs; caches under an older key, from before a synonym change, qualify
only that way).

Each 5 s segment gets the local hour of its midpoint (file start from
birdset.file_local_start, UTC-named files converted). Per site, hour and model:

  minutes          reviewed audio in that hour (only hours with >= 30 min are kept)
  bird_share       share of segments with at least one labelled bird
  species_per_seg  mean labelled species per segment
  n_species        species heard in that hour
  AUROC, cmAP      means over species with positives and negatives in that hour.
                   AUROC does not depend on how common calls are; AP does, so a
                   quiet hour lowers cmAP even if the model is unchanged.
  top1             share of segments with a bird whose top-scoring species is right
  recall, precision, false_alarms_h
                   at a fixed operating point: per species, the threshold reaching
                   90% segment precision over the site's whole test set (no claims
                   if it never does), set before splitting by hour.
                   false_alarms_h = false (segment, species) claims per hour of audio.

Site rows also carry <metric>_site, the site's value over all its hours.
"All" rows (delta=True) give, per hour, the mean over the sites recorded then
of (site's value in that hour - its all-day value): how much better or worse
that time of day is. Pooling the raw values instead would mostly show which
sites record when (05:00 is largely PER, the hardest site). Their activity
columns are pooled over the sites. 95% intervals (<metric>_lo/_hi) resample
recordings within the hour, within each site.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from birdeval import metrics, scoring
from birdeval.metrics import ap, auroc
from birdeval.align import align
from birdeval.cache import PredictionCache
from birdeval.evalsets import build_evalset
from birdeval.evalsets.birdset import file_local_start
from birdeval.models import build_model
from birdeval.recording_metrics import recall_at_precision
from birdeval.runner import alignment_config, load_config
from birdeval.taxonomy import Taxonomy

SITES = ["HSN", "NES", "PER", "POW", "SNE", "SSW", "UHH"]  # NBP has no clock times
TARGET = 0.9
MIN_MINUTES = 30
METRICS = ["AUROC", "cmAP", "top1", "recall", "precision", "false_alarms_h"]


class Block:
    """One site's segments in one hour, for one model: truth, scores, thresholds, recording ids."""

    def __init__(self, y, s, thr, files, seconds):
        self.y, self.s, self.thr, self.files, self.seconds = y, s, thr, files, seconds


def stats(blocks: list[Block]) -> dict:
    aucs, aps, hits, n_top, tp, fp, pos, sec = [], [], 0, 0, 0, 0, 0, 0.0
    for b in blocks:
        y, s = b.y, b.s
        for c in range(y.shape[1]):
            k = y[:, c] != -1
            t = y[k, c]
            if (t == 1).any() and (t == 0).any():
                aucs.append(auroc(t, s[k, c]))
                aps.append(ap(t, s[k, c]))
        rows = np.flatnonzero((y == 1).any(axis=1))
        h = y[rows, np.argmax(s[rows], axis=1)]
        h = h[h != -1]
        hits, n_top = hits + int((h == 1).sum()), n_top + len(h)
        claim = s >= np.where(np.isfinite(b.thr), b.thr, np.inf)
        tp += int((claim & (y == 1)).sum())
        fp += int((claim & (y == 0)).sum())
        pos += int((y == 1).sum())
        sec += b.seconds.sum()
    nan = float("nan")
    return {
        "AUROC": float(np.mean(aucs)) if aucs else nan, "cmAP": float(np.mean(aps)) if aps else nan,
        "n_classes": len(aucs), "top1": hits / n_top if n_top else nan,
        "recall": tp / pos if pos else nan, "precision": tp / (tp + fp) if tp + fp else nan,
        "false_alarms_h": fp / (sec / 3600) if sec else nan,
    }


def resample(b: Block, rng) -> Block:
    ids = np.unique(b.files)
    pick = rng.choice(ids, size=len(ids))
    rows = np.concatenate([np.flatnonzero(b.files == f) for f in pick])
    return Block(b.y[rows], b.s[rows], b.thr, b.files[rows], b.seconds[rows])


def with_ci(point: dict, boots: pd.DataFrame | None) -> dict:
    out = dict(point)
    if boots is not None:
        for m in METRICS:
            out[f"{m}_lo"], out[f"{m}_hi"] = boots[m].quantile([0.025, 0.975]).to_numpy()
    return out


def row(b: Block, n_boot: int) -> tuple[dict, pd.DataFrame | None]:
    """Point values for one site-hour, and its bootstrap draws (recordings resampled)."""
    rng = np.random.default_rng(0)
    boots = pd.DataFrame([stats([resample(b, rng)]) for _ in range(n_boot)]) if n_boot else None
    return stats([b]), boots


def thresholds(y: np.ndarray, s: np.ndarray) -> np.ndarray:
    thr = np.full(y.shape[1], np.nan)
    for c in range(y.shape[1]):
        k = y[:, c] != -1
        thr[c] = recall_at_precision(y[k, c], s[k, c], TARGET)["threshold"]
    return thr


def reference(runs: list[Path], evalset: str, model: str) -> pd.DataFrame | None:
    """The per-class results the report uses for (evalset, model): the last run that has them, as in build_report."""
    found = [f for r in runs if (f := r / evalset / model / "per_class_all.csv").exists()]
    return pd.read_csv(found[-1]).assign(run=found[-1].parts[-4]) if found else None


def matches(y: np.ndarray, s: np.ndarray, classes: list[str], ref: pd.DataFrame) -> bool:
    _, pc = metrics.compute(y, s, classes)
    m = pc.merge(ref, on="class", suffixes=("", "_ref"))
    return len(m) == len(pc) == len(ref) and float(np.abs(m["ap"] - m["ap_ref"]).max()) < 1e-9


def scores(ev, model, cache_root: str, method: str, align_cfg: dict, y: np.ndarray, ref) -> tuple | None:
    """Aligned scores from the first complete cache that reproduces the reference run.

    The cache key hashes the taxonomy synonyms, so a site run before a synonym
    change sits under an older key. Those are tried too (newest first), but only
    used if they give exactly the AP the report shows.
    """
    current = PredictionCache(cache_root, model.cache_key(), ev.name, ev.classes)
    older = sorted((d for d in Path(cache_root).glob(f"{model.name}-*/{current.dir.name}") if d != current.dir),
                   key=lambda d: d.stat().st_mtime, reverse=True)
    for d in [current.dir, *older]:
        c = PredictionCache(cache_root, d.parent.name, ev.name, ev.classes)
        if c.missing(ev.files):
            continue
        windows = {f: scoring.apply(c.get(f), method) for f in ev.files}
        s, _ = align(ev, windows, **align_cfg)
        if ref is None and d == current.dir or ref is not None and matches(y, s, ev.classes, ref):
            return s, d.parent.name
    return None


def site_data(site: str, cfg: dict, tax: Taxonomy, models: dict, tz: str, runs: list[Path]) -> dict | None:
    """{model: (y, s, thr), '_seg': segment frame with hour} for one site, or None if not usable."""
    ev = build_evalset(f"birdset_{site}_scape", cfg["evalsets"][f"birdset_{site}_scape"], tax)
    starts = file_local_start(site, tz)
    if not starts:
        print(f"{site}: no clock times, skipped")
        return None
    cache_root = cfg.get("cache_dir", "cache")
    method = {"method": "logit", **cfg.get("scoring", {})}["method"]
    align_cfg = alignment_config(cfg, method)
    y = ev.target_matrix()
    out, used = {}, []
    for m, model in models.items():
        ref = reference(runs, ev.name, m)
        got = scores(ev, model, cache_root, method, align_cfg, y, ref)
        if got is None:
            short = PredictionCache(cache_root, model.cache_key(), ev.name, ev.classes).missing(ev.files)
            why = (f"{m} missing {len(short)}/{len(ev.files)} files" if short
                   else f"no complete cache of {m} reproduces {ref['run'].iloc[0]}")
            print(f"{site}: skipped, {why}")
            return None
        out[m] = (y, got[0], thresholds(y, got[0]))
        used.append(f"{m} {got[1]}" + (f" = {ref['run'].iloc[0]}" if ref is not None else " (no reference run)"))
    print(f"{site}: {'; '.join(used)}")

    seg = ev.segments[["file", "start", "end"]].copy()
    seg["reviewed"] = (y != -1).any(axis=1)  # gaps between test clips are unreviewed for every species
    clock = seg["file"].map(starts) + (seg["start"] + seg["end"]) / 2
    seg["hour"] = (clock // 3600 % 24).astype("Int64")
    lost = seg["hour"].isna() & seg["reviewed"]
    if lost.any():
        print(f"{site}: {seg.loc[lost, 'file'].nunique()} files without a clock time left out ({lost.sum()} segments)")
    return {**out, "_seg": seg, "_classes": ev.classes}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/eval.yaml")
    ap.add_argument("--sites", nargs="+", default=SITES)
    ap.add_argument("--models", nargs="+", default=["birdnet_2.4", "birdnet_3.0", "perch_v2"])
    ap.add_argument("--out", type=Path, default=Path("results/report/hourly.csv"))
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--runs", nargs="+", type=Path,
                    help="result runs the report is built from, in build_report order (default: results/taxfix_v1 "
                         "results/full_*); a cache is used only if it reproduces their per-class AP")
    args = ap.parse_args()
    cfg = load_config(args.config)
    tax = Taxonomy(**cfg.get("taxonomy", {}))
    models = {m: build_model(m, cfg["models"][m], tax) for m in args.models}  # weights load lazily, never here
    tzs = pd.read_csv("configs/reference/sites.csv").set_index("site")["tz"]

    runs = args.runs or [Path("results/taxfix_v1"), *sorted(Path("results").glob("full_*"))]
    data = {s: d for s in args.sites if (d := site_data(s, cfg, tax, models, tzs[s], runs)) is not None}
    if not data:
        sys.exit("no site has complete cached predictions")
    # each site's all-day values: "All" averages each site's change from these, so an hour's
    # result is not just a matter of which sites happen to record then
    overall = {}
    for site, d in data.items():
        seg = d["_seg"]
        r = np.flatnonzero(seg["hour"].notna().to_numpy() & seg["reviewed"].to_numpy())
        for m in models:
            y, s, thr = d[m]
            overall[site, m] = stats([Block(y[r], s[r], thr, seg["file"].to_numpy()[r],
                                            (seg["end"] - seg["start"]).to_numpy()[r])])
    rows = []
    for hour in range(24):
        pooled = {m: [] for m in models}
        for site, d in data.items():
            seg = d["_seg"]
            r = np.flatnonzero((seg["hour"] == hour).fillna(False).to_numpy() & seg["reviewed"].to_numpy())
            seconds = (seg["end"] - seg["start"]).to_numpy()[r]
            if seconds.sum() < MIN_MINUTES * 60:
                continue
            y0 = d[args.models[0]][0][r]
            known = y0 != -1
            act = {"minutes": seconds.sum() / 60, "n_segments": len(r),
                   "bird_share": float((y0 == 1).any(axis=1).mean()),
                   "species_per_seg": float((y0 == 1).sum(axis=1).mean()),
                   "n_species": int((y0 == 1).any(axis=0).sum()),
                   "reviewed_share": float(known.mean())}
            files = seg["file"].to_numpy()[r]
            for m in models:
                y, s, thr = d[m]
                b = Block(y[r], s[r], thr, files, seconds)
                point, boots = row(b, args.n_boot)
                base = pd.Series({k: overall[site, m][k] for k in METRICS})
                pooled[m].append((site, b, act, pd.Series(point)[METRICS] - base,
                                  None if boots is None else boots[METRICS] - base, point["n_classes"]))
                rows.append({"site": site, "hour": hour, "model": m, "n_files": len(np.unique(files)),
                             **act, **with_ci(point, boots),
                             **{f"{k}_site": v for k, v in base.items()}})
        for m, bs in pooled.items():
            if not bs:
                continue
            acts = pd.DataFrame([x[2] for x in bs])
            n = acts["n_segments"].sum()
            delta = pd.concat([x[3] for x in bs], axis=1).mean(axis=1)  # NaN-skipping mean over sites
            boots = None if bs[0][4] is None else sum(x[4].fillna(0) for x in bs) / sum(x[4].notna() for x in bs)
            rows.append({"site": "All", "hour": hour, "model": m, "delta": True, "sites": " ".join(x[0] for x in bs),
                         "n_files": sum(len(np.unique(x[1].files)) for x in bs),
                         "minutes": acts["minutes"].sum(), "n_segments": int(n),
                         "bird_share": float((acts["bird_share"] * acts["n_segments"]).sum() / n),
                         "species_per_seg": float((acts["species_per_seg"] * acts["n_segments"]).sum() / n),
                         "n_species": int(acts["n_species"].sum()),  # (site, species) pairs
                         "n_classes": int(sum(x[5] for x in bs)),
                         **with_ci(delta.to_dict(), boots)})
        print(f"  hour {hour:02d} done", flush=True)
    out = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.round(5).to_csv(args.out, index=False)
    print(f"wrote {args.out}: {len(data)} sites ({', '.join(data)}), {out[out.site != 'All'].groupby('site').hour.nunique().to_dict()} hours")


if __name__ == "__main__":
    main()

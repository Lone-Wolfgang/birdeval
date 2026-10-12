"""BirdSet label report from metadata only (no audio): label density per site,
and the soundscape-label cross-check.

    python -I scripts/birdset_labels.py META_DIR [--download] [--out results/birdset_labels]
                                        [--reference configs/reference/site_density.csv]

META_DIR holds <SITE>_metadata_test.parquet and <SITE>_metadata_test_5s.parquet
(--download fetches them, ~10 MB in all).

Cross-check: soundscape mode labels each 5 s segment with the official
test_5s clip labels. Here the same segments are re-labelled from the
call-level `test` events with the harness's own event rule
(strong_labels.event_hits), and with a looser any-overlap rule. High
agreement means the clip -> soundscape offset mapping is right; the rule
that agrees best hints at how BirdSet derived its 5 s labels.

--reference also writes the per-site table the report page shows
(recordings, recording length, hours, species, share of clips labelled,
species per recording, exact segment agreement under the harness's rule).
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from birdeval.evalsets.birdset import parse_clip_name
from birdeval.evalsets.strong_labels import event_hits

SITES = ["HSN", "NBP", "NES", "PER", "POW", "SNE", "SSW", "UHH"]
URL = "https://huggingface.co/datasets/DBD-research-group/BirdSet/resolve/data/{site}/{site}_metadata_{split}.parquet"
RULES = {"strong_labels_0.5s": 0.5, "any_overlap": 1e-9}


def _read(meta: Path, site: str, split: str) -> pd.DataFrame:
    df = pd.read_parquet(meta / f"{site}_metadata_{split}.parquet")
    if df.index.name == "filepath":
        df = df.reset_index()
    df["name"] = df["filepath"].map(lambda p: Path(str(p)).name)
    return df


def clips(meta: Path, site: str) -> pd.DataFrame:
    c = _read(meta, site, "test_5s")
    parsed = c["name"].map(parse_clip_name)
    assert parsed.notna().all(), f"{site}: unparseable clip names"
    return pd.DataFrame({
        "scape": parsed.map(lambda p: p[0]),
        "start": parsed.map(lambda p: p[1]),
        "end": parsed.map(lambda p: p[1]) + c["end_time"] - c["start_time"],
        "labels": c["ebird_code_multilabel"].map(lambda a: frozenset(map(str, a))),
    })


def density(c: pd.DataFrame, events: pd.DataFrame, site: str) -> dict:
    n_lab = c["labels"].map(len)
    per_rec = c.groupby("scape")["labels"].agg(lambda s: frozenset().union(*s))
    species = frozenset().union(*per_rec)
    rec_len = c.groupby("scape")["end"].max()
    hours = float((c["end"] - c["start"]).sum() / 3600)
    return {
        "site": site,
        "recordings": len(per_rec),
        "hours": round(hours, 2),
        "rec_len_s_min": float(rec_len.min()),
        "rec_len_s_median": float(rec_len.median()),
        "rec_len_s_max": float(rec_len.max()),
        "clips": len(c),
        "species": len(species),
        "clips_labelled_pct": round(100 * float((n_lab > 0).mean()), 1),
        "labels_per_clip": round(float(n_lab.mean()), 2),
        "species_per_recording": round(float(per_rec.map(len).mean()), 2),
        # recording-level prevalence: share of (recording, site species) pairs present
        "rec_prevalence_pct": round(100 * float(per_rec.map(len).mean() / max(len(species), 1)), 1),
        "calls_per_hour": round(len(events) / hours, 1),
    }


def length_text(lo: float, mid: float, hi: float) -> str:
    """Recording length for people: '1 h', '300 s', or a range when lengths vary."""
    def one(x):
        return f"{x / 3600:g} h" if x >= 3600 - 10 else f"{round(x):g} s"
    return one(mid) if hi - lo <= 0.01 * mid else f"{round(lo):g}–{round(hi):g} s"


def reference_table(dens: pd.DataFrame, checks: pd.DataFrame) -> pd.DataFrame:
    exact = checks[checks["rule"] == "strong_labels_0.5s"].set_index("site")["segment_exact_match_pct"]
    return pd.DataFrame({
        "site": dens["site"],
        "recordings": dens["recordings"],
        "length": [length_text(a, b, c) for a, b, c in
                   zip(dens["rec_len_s_min"], dens["rec_len_s_median"], dens["rec_len_s_max"])],
        "hours": dens["hours"].round(2),
        "species": dens["species"],
        "share_labelled": (dens["clips_labelled_pct"] / 100).round(3),
        "species_per_recording": dens["species_per_recording"].round(2),
        "exact_agreement": (dens["site"].map(exact) / 100).round(3),
    })


def crosscheck(c: pd.DataFrame, events: pd.DataFrame, min_overlap: float) -> dict:
    ev_by = {k: g for k, g in events.groupby("name")}
    exact, b_pos, c_pos, both, cells, cell_agree = 0, 0, 0, 0, 0, 0
    species = frozenset().union(*c["labels"]) | frozenset(events["ebird_code"].astype(str))
    for scape, g in c.groupby("scape"):
        ev = ev_by.get(scape, events.iloc[:0])
        for s, e, b in zip(g["start"], g["end"], g["labels"]):
            hit = event_hits(ev["start_time"], ev["end_time"], s, e, min_overlap) if len(ev) else np.zeros(0, bool)
            lab = frozenset(ev["ebird_code"].astype(str)[hit])
            exact += lab == b
            b_pos, c_pos, both = b_pos + len(b), c_pos + len(lab), both + len(b & lab)
            cells += len(species)
            cell_agree += len(species) - len(b ^ lab)
    return {
        "segment_exact_match_pct": round(100 * exact / len(c), 2),
        "positive_jaccard": round(both / max(b_pos + c_pos - both, 1), 4),
        "official_only": b_pos - both,
        "events_only": c_pos - both,
        "cell_agreement_pct": round(100 * cell_agree / max(cells, 1), 3),
    }


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("meta_dir")
    p.add_argument("--download", action="store_true")
    p.add_argument("--out", default="results/birdset_labels")
    p.add_argument("--reference", help="also write the report's site table here (configs/reference/site_density.csv)")
    a = p.parse_args(argv)
    meta, out = Path(a.meta_dir), Path(a.out)
    meta.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)

    dens, checks = [], []
    for site in SITES:
        for split in ("test", "test_5s"):
            f = meta / f"{site}_metadata_{split}.parquet"
            if a.download and not f.exists():
                urllib.request.urlretrieve(URL.format(site=site, split=split), f)
        c, events = clips(meta, site), _read(meta, site, "test")
        dens.append(density(c, events, site))
        for rule, ov in RULES.items():
            checks.append({"site": site, "rule": rule, **crosscheck(c, events, ov)})

    dens, checks = pd.DataFrame(dens), pd.DataFrame(checks)
    dens.to_csv(out / "density.csv", index=False)
    checks.to_csv(out / "crosscheck.csv", index=False)
    if a.reference:
        ref = Path(a.reference)
        with ref.open("w") as fh:
            fh.write("# Source: scripts/birdset_labels.py (BirdSet metadata only, no audio). "
                     "exact_agreement: share of 5 s segments whose labels rebuilt from call annotations "
                     "(>= 0.5 s overlap) equal the official labels.\n")
            reference_table(dens, checks).to_csv(fh, index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(dens.to_string(index=False), "\n")
        print(checks.to_string(index=False))


if __name__ == "__main__":
    main()

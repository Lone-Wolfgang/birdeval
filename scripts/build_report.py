"""Tables for the evaluation report: models x sites, published check, and
per-species AP grouped by habitat and by label abundance.

    python -I scripts/build_report.py RUN_DIR [RUN_DIR ...] [--out results/report]

Writes CSVs to --out, one per report section:
  headline.csv     site x model: segment cmAP (95% CI), AUROC, top-1 from clips
                   mode, and recording-level cmAP from soundscape mode
  published.csv    Perch V2 against the Perch 2.0 paper (scripts/compare_published.py)
  by_abundance.csv mean per-species AP per model and label-abundance bin
  by_habitat.csv   the same per AVONET habitat (needs configs/reference/species_traits.csv,
                   from scripts/avonet_traits.py; skipped without it)

All metrics use the `all` scope (BirdSet's protocol). A group's unit is a
(site, species) pair; its CI resamples those pairs (2000 draws, seed 0).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from compare_published import compare, load_summaries, ours  # noqa: E402

SITE = re.compile(r"^birdset_([A-Z]{3})(_scape)?$")
TRAITS = Path("configs/reference/species_traits.csv")
# Bins fixed before looking at results: labelled 5 s clips per species and site.
ABUNDANCE = [(1, 9, "1–9 clips"), (10, 49, "10–49 clips"), (50, None, "50+ clips")]
MIN_GROUP = 5  # habitats with fewer (site, species) pairs merge into "Other"


def summaries(runs: list[Path]) -> pd.DataFrame:
    df = load_summaries(runs)
    m = df["evalset"].str.extract(SITE)
    df["site"], df["scape"] = m[0], m[1].notna()
    return df[df["site"].notna() & (df["classes"] == "all")]


def headline(s: pd.DataFrame) -> pd.DataFrame:
    clips = s[~s["scape"]]
    rec = s[s["scape"]][["site", "model", "rec_cmAP"]]
    out = clips.drop(columns="rec_cmAP").merge(rec, on=["site", "model"], how="left")
    cols = ["site", "model", "cmAP", "cmAP_lo", "cmAP_hi", "AUROC", "top1", "rec_cmAP", "n_classes_scored", "n_segments"]
    return out[cols].rename(columns={"n_classes_scored": "n_species", "n_segments": "n_clips"}).round(4)


def per_class(runs: list[Path]) -> pd.DataFrame:
    frames = []
    for d in runs:
        for f in d.glob("birdset_*/*/per_class_all.csv"):
            m = SITE.match(f.parent.parent.name)
            if m and not m.group(2):  # clips mode only
                frames.append(pd.read_csv(f).assign(site=m.group(1), model=f.parent.name))
    return pd.concat(frames, ignore_index=True).rename(columns={"class": "species"})


def grouped(pc: pd.DataFrame, group: pd.Series, order: list[str], n_boot: int = 2000) -> pd.DataFrame:
    """Mean AP per (model, group) over (site, species) pairs, CI by resampling pairs."""
    pc = pc.assign(group=group)
    rng = np.random.default_rng(0)
    rows = []
    for (model, g), sub in pc.groupby(["model", "group"]):
        ap = sub["ap"].to_numpy()
        boots = rng.choice(ap, size=(n_boot, len(ap))).mean(axis=1)
        rows.append({
            "model": model, "group": g, "n_species": len(ap), "mean_ap": ap.mean(),
            "ap_lo": np.quantile(boots, 0.025), "ap_hi": np.quantile(boots, 0.975),
            "sites": ", ".join(sorted(sub["site"].unique())),
        })
    out = pd.DataFrame(rows)
    out["order"] = out["group"].map({g: i for i, g in enumerate(order)})
    return out.sort_values(["order", "model"]).drop(columns="order").round(4)


def abundance_bin(n: pd.Series) -> pd.Series:
    out = pd.Series(index=n.index, dtype=object)
    for lo, hi, name in ABUNDANCE:
        out[(n >= lo) & ((n <= hi) if hi else True)] = name
    return out


def habitat(pc: pd.DataFrame) -> tuple[pd.Series, list[str]] | None:
    if not TRAITS.exists():
        return None
    traits = pd.read_csv(TRAITS).set_index("species")["habitat"]
    h = pc["species"].map(traits).fillna("Unknown")
    # pairs per habitat for one model: every model scores the same pairs
    counts = h[pc["model"] == pc["model"].iloc[0]].value_counts()
    small = counts[counts < MIN_GROUP].index
    h = h.where(~h.isin(small), "Other")
    order = [g for g in counts.index if g not in small] + ["Other", "Unknown"]
    return h, order


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=Path("results/report"))
    ap.add_argument("--ref", type=Path, default=Path("configs/reference/published.yaml"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    head = headline(summaries(args.runs))
    head.to_csv(args.out / "headline.csv", index=False)

    pub = compare(ours(args.runs), yaml.safe_load(args.ref.read_text()), tol=0.02)
    pub = pub[pub["pub_lo"].notna()]
    pub[["site", "model", "metric", "ours", "pub_lo", "pub_hi", "delta", "flag"]].round(4).to_csv(
        args.out / "published.csv", index=False)

    pc = per_class(args.runs)
    # every model must score the same (site, species) pairs, or groups aren't comparable
    pairs = pc.groupby("model").apply(lambda g: frozenset(zip(g["site"], g["species"])), include_groups=False)
    assert pairs.nunique() == 1, "models scored different species"
    grouped(pc, abundance_bin(pc["n_pos"]), [b[2] for b in ABUNDANCE]).to_csv(
        args.out / "by_abundance.csv", index=False)

    hab = habitat(pc)
    if hab is None:
        print(f"no {TRAITS}: skipped by_habitat.csv (run scripts/avonet_traits.py)")
    else:
        grouped(pc, *hab).to_csv(args.out / "by_habitat.csv", index=False)
    print(f"wrote {', '.join(sorted(p.name for p in args.out.glob('*.csv')))} to {args.out}")


if __name__ == "__main__":
    main()

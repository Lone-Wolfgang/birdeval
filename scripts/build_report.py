"""Tables for the evaluation report: models x sites, published check, and
per-species results that the page groups, filters and sorts.

    python -I scripts/build_report.py RUN_DIR [RUN_DIR ...] [--out results/report]

Writes CSVs to --out:
  headline.csv  mode x site x model: segment cmAP (95% CI), AUROC, top-1, and
                for soundscapes recording-level cmAP and recall at 90% precision.
                Site "All" is the mean over sites that every model has finished.
  published.csv Perch V2 against the Perch 2.0 paper (scripts/compare_published.py)
  species.csv   one row per mode x site x species: n_pos, traits, site biome,
                and each model's AP, AUROC and recall at 90% precision
                (soundscapes add recording-level AP)
  groups.csv    mean per-species AP per mode x scope (All or one site) x
                grouping (habitat, biome, prevalence) x group x model

Mode is clips (birdset_<SITE>, BirdSet's protocol) or scape (birdset_<SITE>_scape).
All metrics use the `all` scope. A group's unit is a (site, species) pair; its
CI resamples those pairs (2000 draws, seed 0). Groups use only sites that every
model has finished in that mode, so models are compared on the same pairs.
If two runs hold the same (mode, site, model), the later RUN_DIR wins.
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
SITES = Path("configs/reference/sites.csv")
# Bins fixed before looking at results: labelled 5 s segments per species and site.
ABUNDANCE = [(1, 9, "1–9 segments"), (10, 49, "10–49 segments"), (50, None, "50+ segments")]
MIN_GROUP = 5  # habitats with fewer (site, species) pairs merge into "Other"
KEY = ["mode", "site", "model"]


def site_mode(evalset: pd.Series) -> pd.DataFrame:
    m = evalset.str.extract(SITE)
    return pd.DataFrame({"site": m[0], "mode": np.where(m[1].notna(), "scape", "clips")}, index=evalset.index)


def summaries(runs: list[Path]) -> pd.DataFrame:
    df = load_summaries(runs)
    df[["site", "mode"]] = site_mode(df["evalset"])
    df = df[df["site"].notna() & (df["classes"] == "all")].copy()
    # in clips mode each clip is its own recording unit, so recording cmAP repeats cmAP
    df.loc[df["mode"] == "clips", "rec_cmAP"] = np.nan
    return df.drop_duplicates(KEY, keep="last")


def headline(s: pd.DataFrame) -> pd.DataFrame:
    cols = ["cmAP", "cmAP_lo", "cmAP_hi", "AUROC", "top1", "rec_cmAP", "rec_recall90"]
    out = s[KEY + cols + ["n_classes_scored", "n_segments"]].rename(
        columns={"n_classes_scored": "n_species", "n_segments": "n_segs"})
    rows = []
    for mode, sub in out.groupby("mode"):
        done = complete_sites(sub)
        sub = sub[sub["site"].isin(done)]
        for model, g in sub.groupby("model"):
            rows.append({"mode": mode, "site": "All", "model": model, **g[cols].mean().to_dict(),
                         "cmAP_lo": np.nan, "cmAP_hi": np.nan,
                         "n_species": g["n_species"].sum(), "n_segs": g["n_segs"].sum(),
                         "sites": " ".join(sorted(done))})
    return pd.concat([out, pd.DataFrame(rows)], ignore_index=True).round(4)


def complete_sites(df: pd.DataFrame) -> list[str]:
    """Sites where every model in df has a result."""
    n = df["model"].nunique()
    per = df.groupby("site")["model"].nunique()
    return sorted(per[per == n].index)


def per_class(runs: list[Path]) -> pd.DataFrame:
    """Long table: one row per run x mode x site x model x species."""
    frames = []
    for d in runs:
        for f in sorted(d.glob("birdset_*/*/per_class_all.csv")):
            m = SITE.match(f.parent.parent.name)
            if not m:
                continue
            pc = pd.read_csv(f).rename(columns={"class": "species"})
            rec_f = f.with_name("per_species_recording_all.csv")
            if rec_f.exists():
                rec = pd.read_csv(rec_f)[["species", "n_present", "ap", "recall_at_p90"]].rename(
                    columns={"ap": "rec_ap", "recall_at_p90": "recall90"})
                if not m.group(2):  # clips are their own units: rec AP is AP, n_present is n_pos
                    rec = rec.drop(columns=["rec_ap", "n_present"])
                pc = pc.merge(rec, on="species", how="left")
            frames.append(pc.assign(mode="scape" if m.group(2) else "clips", site=m.group(1), model=f.parent.name))
    pc = pd.concat(frames, ignore_index=True)
    return pc.drop_duplicates(KEY + ["species"], keep="last")


def species_table(pc: pd.DataFrame, traits: pd.DataFrame, sites: pd.DataFrame) -> pd.DataFrame:
    """Wide table: one row per mode x site x species, one column per metric x model."""
    idx = ["mode", "site", "species"]
    metrics = [c for c in ["ap", "auroc", "rec_ap", "recall90"] if c in pc]
    wide = pc.set_index(idx + ["model"])[metrics].unstack("model")
    wide.columns = [f"{metric}_{model}" for metric, model in wide.columns]
    wide = wide.dropna(axis=1, how="all")
    counts = pc.groupby(idx)[[c for c in ["n_pos", "n_present"] if c in pc]].max()
    out = counts.join(wide).reset_index()
    out = out.merge(traits[["species", "family", "habitat", "migration"]], on="species", how="left")
    out = out.merge(sites[["site", "biome"]], on="site", how="left")
    return out.sort_values(idx).round(4)


def grouped(pc: pd.DataFrame, group: pd.Series, order: list[str], n_boot: int = 2000) -> pd.DataFrame:
    """Mean AP per (model, group) over (site, species) pairs, CI by resampling pairs."""
    pc = pc.assign(group=group)
    rng = np.random.default_rng(0)
    rows = []
    for (model, g), sub in pc.groupby(["model", "group"]):
        ap = sub["ap"].to_numpy()
        boots = rng.choice(ap, size=(n_boot, len(ap))).mean(axis=1)
        rows.append({
            "model": model, "group": g, "n_pairs": len(ap), "mean_ap": ap.mean(),
            "ap_lo": np.quantile(boots, 0.025), "ap_hi": np.quantile(boots, 0.975),
            "sites": " ".join(sorted(sub["site"].unique())),
        })
    out = pd.DataFrame(rows)
    out["order"] = out["group"].map({g: i for i, g in enumerate(order)})
    return out.sort_values(["order", "model"]).drop(columns="order").round(4)


def abundance_bin(n: pd.Series) -> pd.Series:
    out = pd.Series(index=n.index, dtype=object)
    for lo, hi, name in ABUNDANCE:
        out[(n >= lo) & ((n <= hi) if hi else True)] = name
    return out


def habitat(pc: pd.DataFrame, traits: pd.Series) -> tuple[pd.Series, list[str]]:
    h = pc["species"].map(traits).fillna("Unknown")
    # pairs per habitat for one model: every model scores the same pairs
    counts = h[pc["model"] == pc["model"].iloc[0]].value_counts()
    small = counts[counts < MIN_GROUP].index
    h = h.where(~h.isin(small), "Other")
    order = [g for g in counts.index if g not in small] + ["Other", "Unknown"]
    return h, order


def groups_table(pc: pd.DataFrame, traits: pd.DataFrame, sites: pd.DataFrame) -> pd.DataFrame:
    hab = traits.set_index("species")["habitat"]
    biome = sites.set_index("site")["biome"]
    frames = []
    for mode, m in pc.groupby("mode"):
        done = complete_sites(m)
        skipped = sorted(set(m["site"]) - set(done))
        if skipped:
            print(f"{mode}: {', '.join(skipped)} left out of groups (not every model has finished)")
        m = m[m["site"].isin(done)]
        for scope in ["All", *done]:
            sub = m if scope == "All" else m[m["site"] == scope]
            # every model must score the same (site, species) pairs, or groups aren't comparable
            pairs = sub.groupby("model").apply(lambda g: frozenset(zip(g["site"], g["species"])), include_groups=False)
            assert pairs.nunique() == 1, f"{mode}/{scope}: models scored different species"
            parts = {
                "prevalence": (abundance_bin(sub["n_pos"]), [b[2] for b in ABUNDANCE]),
                "habitat": habitat(sub, hab),
            }
            if scope == "All":  # one site is one biome
                b = sub["site"].map(biome).fillna("Unknown")
                parts["biome"] = (b, list(dict.fromkeys(biome[biome.index.isin(done)])) + ["Unknown"])
            for name, (g, order) in parts.items():
                frames.append(grouped(sub, g, order).assign(mode=mode, scope=scope, grouping=name))
    out = pd.concat(frames, ignore_index=True)
    return out[["mode", "scope", "grouping", "group", "model", "n_pairs", "mean_ap", "ap_lo", "ap_hi", "sites"]]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=Path("results/report"))
    ap.add_argument("--ref", type=Path, default=Path("configs/reference/published.yaml"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    headline(summaries(args.runs)).to_csv(args.out / "headline.csv", index=False)

    pub = compare(ours(args.runs), yaml.safe_load(args.ref.read_text()), tol=0.02)
    pub = pub[pub["pub_lo"].notna()]
    pub[["site", "model", "metric", "ours", "pub_lo", "pub_hi", "delta", "flag"]].round(4).to_csv(
        args.out / "published.csv", index=False)

    if not TRAITS.exists():
        sys.exit(f"no {TRAITS}: run scripts/avonet_traits.py first")
    traits, sites = pd.read_csv(TRAITS), pd.read_csv(SITES)
    pc = per_class(args.runs)
    species_table(pc, traits, sites).to_csv(args.out / "species.csv", index=False)
    groups_table(pc, traits, sites).to_csv(args.out / "groups.csv", index=False)
    print(f"wrote {', '.join(sorted(p.name for p in args.out.glob('*.csv')))} to {args.out}")


if __name__ == "__main__":
    main()

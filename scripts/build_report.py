"""Tables for the evaluation report: models x sites, published check, and
per-species results that the page groups, filters and sorts.

    python -I scripts/build_report.py RUN_DIR [RUN_DIR ...] [--out results/report]

Writes CSVs to --out:
  headline.csv  mode x site x model: segment cmAP (95% CI), AUROC, top-1, and
                for soundscapes recording-level cmAP, recall at 90% precision and
                evidence precision at that threshold (all claims, and claims right
                at recording level).
                Site "All" is the mean over sites that every model has finished.
  published.csv Perch V2 against the Perch 2.0 paper (scripts/compare_published.py)
  species.csv   one row per mode x site x species: n_pos, each model's AP,
                AUROC and recall at 90% precision (soundscapes add
                recording-level AP), and what is known about the species
                (see species_info)
  groups.csv    mean per-species AP per mode x scope (All or one site) x
                grouping x group x model; groupings are in GROUPINGS
  weakest.csv   the WEAKEST_N clips-mode (site, species) pairs with at least
                WEAKEST_MIN_POS labelled segments, ranked by the best AP any model
                gets (low = every model struggles). Rule fixed before looking.

Species information comes from configs/ (each is optional except AVONET):
  taxonomy/ebird_taxonomy.csv     common name, eBird code
  reference/species_traits.csv    AVONET: habitat, mass, lifestyle, diet niche (scripts/avonet_traits.py)
  reference/elton_traits.csv      EltonTraits: diet, foraging stratum, nocturnal (scripts/elton_traits.py)
  reference/site_calls.csv        per site: call frequency band, calls per hour (scripts/site_calls.py)
  reference/xc_counts.csv         xeno-canto recordings (scripts/xc_counts.py)

Mode is clips (birdset_<SITE>, BirdSet's protocol) or scape (birdset_<SITE>_scape).
All metrics use the `all` scope. A group's unit is a (site, species) pair; its
CI resamples those pairs (2000 draws, seed 0). Groups use only sites that every
model has finished in that mode, so models are compared on the same pairs.
If two runs hold the same (mode, site, model), the later RUN_DIR wins.

Only public BirdSet eval sets (names starting birdset_) are read; any other
eval set in the run folders is skipped with a warning. --allow-nonpublic keeps
them (site = eval set name), for local use only: render_report.py refuses to
write such tables into docs/.
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

from birdeval.taxonomy import normalize_sci  # noqa: E402

SITE = re.compile(r"^birdset_([A-Z]{3})(_scape)?$")
PUBLIC = "birdset_"
REF = Path("configs/reference")
TRAITS = REF / "species_traits.csv"
SITES = REF / "sites.csv"
EBIRD = Path("configs/taxonomy/ebird_taxonomy.csv")
# Bins fixed before looking at results, as [lo, hi): labelled 5 s segments per
# species and site; AVONET body mass; xeno-canto recordings; median call centre frequency.
ABUNDANCE = [(1, 10, "1–9 segments"), (10, 50, "10–49 segments"), (50, None, "50+ segments")]
MASS = [(0, 20, "Under 20 g"), (20, 100, "20–100 g"), (100, 500, "100–500 g"), (500, None, "500 g and over")]
XC = [(0, 100, "Under 100 recordings"), (100, 500, "100–499"), (500, 2000, "500–1,999"), (2000, None, "2,000 and over")]
FREQ = [(0, 2, "Under 2 kHz"), (2, 4, "2–4 kHz"), (4, 8, "4–8 kHz"), (8, None, "8 kHz and over")]
# grouping -> (species_info column, bins or None for categories)
GROUPINGS = {
    "prevalence": ("n_pos", ABUNDANCE),
    "habitat": ("habitat", None),
    "biome": ("biome", None),
    "mass": ("mass_g", MASS),
    "lifestyle": ("lifestyle", None),
    "stratum": ("stratum", None),
    "diet": ("diet", None),
    "call_freq": ("freq_centre_khz", FREQ),
    "xc": ("xc_total", XC),
}
WEAKEST_N, WEAKEST_MIN_POS = 20, 10
MIN_GROUP = 5  # categories with fewer (site, species) pairs merge into "Other"
KEY = ["mode", "site", "model"]


def public_only(evalsets: pd.Series, allow_nonpublic: bool) -> pd.Series:
    """Mask of eval sets to keep; warns once about each non-public one."""
    public = evalsets.str.startswith(PUBLIC)
    for e in sorted(set(evalsets[~public])):
        print(f"warning: {e} is not a public BirdSet eval set: "
              + ("kept (--allow-nonpublic)" if allow_nonpublic else "skipped"))
    return public | allow_nonpublic


def site_mode(evalset: pd.Series) -> pd.DataFrame:
    m = evalset.str.extract(SITE)
    scape = m[1].notna() | evalset.str.endswith("_scape")
    # non-BirdSet sets (kept only with --allow-nonpublic) are their own "site";
    # other birdset_ sets (e.g. smoke subsets) have none and are dropped
    site = m[0].where(evalset.str.startswith(PUBLIC), evalset.str.removesuffix("_scape"))
    return pd.DataFrame({"site": site, "mode": np.where(scape, "scape", "clips")}, index=evalset.index)


def summaries(runs: list[Path], allow_nonpublic: bool = False) -> pd.DataFrame:
    df = load_summaries(runs)
    df = df[public_only(df["evalset"], allow_nonpublic)].copy()
    df[["site", "mode"]] = site_mode(df["evalset"])
    df = df[df["site"].notna() & (df["classes"] == "all")].copy()
    # in clips mode each clip is its own recording unit, so recording cmAP repeats cmAP;
    # evidence precision is a soundscape question (which window in a recording)
    df.loc[df["mode"] == "clips", ["rec_cmAP", "rec_evidence90", "rec_evidence90_cond"]] = np.nan
    return df.drop_duplicates(KEY, keep="last")


def headline(s: pd.DataFrame) -> pd.DataFrame:
    cols = ["cmAP", "cmAP_lo", "cmAP_hi", "AUROC", "top1", "rec_cmAP", "rec_recall90",
            "rec_evidence90", "rec_evidence90_cond"]
    out = s[KEY + ["evalset"] + cols + ["n_classes_scored", "n_segments"]].rename(
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


def per_class(runs: list[Path], allow_nonpublic: bool = False) -> pd.DataFrame:
    """Long table: one row per run x mode x site x model x species."""
    frames = []
    for d in runs:
        files = sorted(d.glob("*/*/per_class_all.csv"))
        names = pd.Series([f.parent.parent.name for f in files], dtype=str)
        keep = public_only(names, allow_nonpublic) if len(files) else []
        for f, ok in zip(files, keep):
            if not ok:
                continue
            evalset = f.parent.parent.name
            sm = site_mode(pd.Series([evalset])).iloc[0]
            if pd.isna(sm["site"]):
                continue
            pc = pd.read_csv(f).rename(columns={"class": "species"})
            rec_f = f.with_name("per_species_recording_all.csv")
            if rec_f.exists():
                rec = pd.read_csv(rec_f)[["species", "n_present", "ap", "recall_at_p90"]].rename(
                    columns={"ap": "rec_ap", "recall_at_p90": "recall90"})
                if sm["mode"] == "clips":  # clips are their own units: rec AP is AP, n_present is n_pos
                    rec = rec.drop(columns=["rec_ap", "n_present"])
                pc = pc.merge(rec, on="species", how="left")
            frames.append(pc.assign(mode=sm["mode"], site=sm["site"], model=f.parent.name, evalset=evalset))
    pc = pd.concat(frames, ignore_index=True)
    return pc.drop_duplicates(KEY + ["species"], keep="last")


def species_info(pairs: pd.DataFrame, sites: pd.DataFrame) -> pd.DataFrame:
    """Everything known about each (site, species) besides its scores; missing sources are skipped."""
    out = pairs[["site", "species"]].drop_duplicates().merge(sites[["site", "biome"]], on="site", how="left")
    if EBIRD.exists():
        eb = pd.read_csv(EBIRD)
        cols = {c.upper(): c for c in eb.columns}
        sci = cols.get("SCI_NAME") or cols.get("SCIENTIFIC_NAME")
        com = cols.get("PRIMARY_COM_NAME") or cols.get("COMMON_NAME")
        eb = pd.DataFrame({"species": eb[sci].map(normalize_sci), "common_name": eb[com],
                           "ebird_code": eb[cols["SPECIES_CODE"]]}).drop_duplicates("species")
        out = out.merge(eb, on="species", how="left")
    traits = pd.read_csv(TRAITS)
    out = out.merge(traits[[c for c in ["species", "family", "habitat", "migration", "habitat_density", "mass_g",
                                        "wing_mm", "trophic_niche", "lifestyle"] if c in traits]],
                    on="species", how="left")
    if (f := REF / "elton_traits.csv").exists():
        out = out.merge(pd.read_csv(f)[["species", "diet", "stratum", "nocturnal"]], on="species", how="left")
    if (f := REF / "site_calls.csv").exists():
        calls = pd.read_csv(f)
        hours = [f"h{h:02d}" for h in range(24)]
        calls["hours"] = calls[hours].astype(int).astype(str).agg(";".join, axis=1)
        out = out.merge(calls[["site", "species", "n_events", "freq_low_khz", "freq_high_khz", "freq_centre_khz",
                               "night_share", "hours"]], on=["site", "species"], how="left")
    if (f := REF / "xc_counts.csv").exists():
        out = out.merge(pd.read_csv(f)[["species", "site", "xc_total", "xc_site_countries"]],
                        on=["site", "species"], how="left")
    return out


def species_table(pc: pd.DataFrame, info: pd.DataFrame) -> pd.DataFrame:
    """Wide table: one row per mode x site x species, one column per metric x model."""
    idx = ["mode", "site", "species"]
    metrics = [c for c in ["ap", "auroc", "rec_ap", "recall90"] if c in pc]
    wide = pc.set_index(idx + ["model"])[metrics].unstack("model")
    wide.columns = [f"{metric}_{model}" for metric, model in wide.columns]
    wide = wide.dropna(axis=1, how="all")
    counts = pc.groupby(idx)[[c for c in ["n_pos", "n_present"] if c in pc]].max()
    counts["evalset"] = pc.groupby(idx)["evalset"].first()
    out = counts.join(wide).reset_index().merge(info, on=["site", "species"], how="left")
    return out.sort_values(idx).round(4)


def weakest(species: pd.DataFrame, n: int = WEAKEST_N, min_pos: int = WEAKEST_MIN_POS) -> pd.DataFrame:
    """Clips mode, scope all: the (site, species) pairs where even the best model's AP is lowest."""
    c = species[(species["mode"] == "clips") & (species["n_pos"] >= min_pos)].copy()
    aps = [col for col in c.columns if col.startswith("ap_")]
    c["max_ap"] = c[aps].max(axis=1)
    keep = ["site", "species", "common_name", "ebird_code", "family", "habitat", "n_pos", "xc_total", *aps, "max_ap"]
    return c.sort_values(["max_ap", "site", "species"])[[k for k in keep if k in c]].head(n).reset_index(drop=True)


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


def binned(x: pd.Series, bins: list[tuple]) -> tuple[pd.Series, list[str]]:
    out = pd.Series("No data", index=x.index, dtype=object)
    for lo, hi, name in bins:
        out[(x >= lo) & ((x < hi) if hi is not None else True)] = name
    return out, [b[2] for b in bins] + ["No data"]


def categories(sub: pd.DataFrame, x: pd.Series) -> tuple[pd.Series, list[str]]:
    x = x.fillna("Unknown").astype(str).replace({"True": "Nocturnal", "False": "Diurnal"})
    # pairs per category for one model: every model scores the same pairs
    counts = x[sub["model"] == sub["model"].iloc[0]].value_counts()
    small = counts[(counts < MIN_GROUP) & (counts.index != "Unknown")].index  # unknown stays unknown
    x = x.where(~x.isin(small), "Other")
    order = [g for g in counts.index if g not in small and g != "Unknown"] + ["Other", "Unknown"]
    return x, order


def groups_table(pc: pd.DataFrame, info: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for mode, m in pc.groupby("mode"):
        done = complete_sites(m)
        skipped = sorted(set(m["site"]) - set(done))
        if skipped:
            print(f"{mode}: {', '.join(skipped)} left out of groups (not every model has finished)")
        m = m[m["site"].isin(done)].merge(info, on=["site", "species"], how="left", suffixes=("", "_info"))
        for scope in ["All", *done]:
            sub = m if scope == "All" else m[m["site"] == scope]
            # every model must score the same (site, species) pairs, or groups aren't comparable
            pairs = sub.groupby("model").apply(lambda g: frozenset(zip(g["site"], g["species"])), include_groups=False)
            assert pairs.nunique() == 1, f"{mode}/{scope}: models scored different species"
            for name, (col, bins) in GROUPINGS.items():
                if col not in sub or (name == "biome" and scope != "All"):  # one site is one biome
                    continue
                if sub[col].isna().all():
                    continue
                g, order = binned(sub[col], bins) if bins else categories(sub, sub[col])
                frames.append(grouped(sub, g, order).assign(mode=mode, scope=scope, grouping=name))
    out = pd.concat(frames, ignore_index=True)
    return out[["mode", "scope", "grouping", "group", "model", "n_pairs", "mean_ap", "ap_lo", "ap_hi", "sites"]]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=Path("results/report"))
    ap.add_argument("--ref", type=Path, default=Path("configs/reference/published.yaml"))
    ap.add_argument("--allow-nonpublic", action="store_true",
                    help="keep eval sets that aren't public BirdSet sets (local use only; never publish)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    headline(summaries(args.runs, args.allow_nonpublic)).to_csv(args.out / "headline.csv", index=False)

    pub = compare(ours(args.runs), yaml.safe_load(args.ref.read_text()), tol=0.02)
    pub = pub[pub["pub_lo"].notna()]
    pub[["site", "model", "metric", "ours", "pub_lo", "pub_hi", "delta", "flag"]].round(4).to_csv(
        args.out / "published.csv", index=False)

    if not TRAITS.exists():
        sys.exit(f"no {TRAITS}: run scripts/avonet_traits.py first")
    pc = per_class(args.runs, args.allow_nonpublic)
    info = species_info(pc, pd.read_csv(SITES))
    sp = species_table(pc, info)
    sp.to_csv(args.out / "species.csv", index=False)
    weakest(sp).to_csv(args.out / "weakest.csv", index=False)
    groups_table(pc, info).to_csv(args.out / "groups.csv", index=False)
    print(f"wrote {', '.join(sorted(p.name for p in args.out.glob('*.csv')))} to {args.out}")


if __name__ == "__main__":
    main()

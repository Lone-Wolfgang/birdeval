"""Prose templates for the report page: data tokens, generated blocks and repo-file links.

Section files (report/sections/*.html) are plain HTML fragments with three kinds of
placeholder:

    {{token}}  {{token|fmt}}   a value from the facts dict; fmt is 3f, 2f, pct, int or
                               range (a (lo, hi) pair rendered "a–b" to 2 decimals)
    {{block:name}}             generated HTML (a table, a figure, the explorer)
    {{file:path/in/repo}}      a repo file: a link when repo_url is set, else code text

An unknown token, block or format is an error, so a typo can't publish as text.
Facts that no template uses are reported as warnings.
"""

from __future__ import annotations

import re
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd

PLACEHOLDER = re.compile(r"\{\{\s*([^{}|]+?)\s*(?:\|\s*([^{}]+?)\s*)?\}\}")
MODEL_KEYS = {"birdnet_2.4": "b24", "birdnet_3.0": "b30", "perch_v2": "perch"}
BIOME_KEYS = {
    "Tropical moist broadleaf forest": "tropical",
    "Temperate broadleaf & mixed forest": "temperate_broadleaf",
    "Temperate conifer forest": "temperate_conifer",
    "Mediterranean forest, woodland & scrub": "mediterranean",
}
XC_KEYS = {"Under 100 recordings": "under100", "100–499": "100_499", "500–1,999": "500_1999",
           "2,000 and over": "2000plus", "No data": "nodata"}
PREV_KEYS = {"1–9 segments": "1_9", "10–49 segments": "10_49", "50+ segments": "50plus"}


class TemplateError(ValueError):
    pass


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def fmt(value, how: str | None) -> str:
    """Format one fact. Strings are HTML-escaped; numbers follow `how`."""
    if how is None:
        if isinstance(value, str):
            return escape(value)
        if isinstance(value, (int, np.integer)):
            how = "int"
        elif isinstance(value, (float, np.floating)):
            how = "3f"
        elif isinstance(value, tuple):
            how = "range"
        else:
            raise TemplateError(f"no default format for {type(value).__name__}")
    if how == "range":
        lo, hi = value
        return f"{lo:.2f}–{hi:.2f}"
    if isinstance(value, str):
        raise TemplateError(f"format {how!r} on text {value!r}")
    if value is None or (isinstance(value, float) and np.isnan(value)):
        raise TemplateError("missing value")
    if how == "3f":
        return f"{value:.3f}"
    if how == "2f":
        return f"{value:.2f}"
    if how == "pct":
        return f"{round(100 * value)}%"
    if how == "int":
        return f"{int(round(value)):,}"
    raise TemplateError(f"unknown format {how!r}")


def fill(text: str, facts: dict, blocks: dict, repo_url: str = "", used: set | None = None,
         where: str = "") -> str:
    """Replace every placeholder in text. Records the fact names it used in `used`."""

    def one(m: re.Match) -> str:
        name, how = m.group(1), m.group(2)
        try:
            if name.startswith("block:"):
                key = name[6:]
                if key not in blocks:
                    raise TemplateError(f"unknown block {key!r}")
                return blocks[key]
            if name.startswith("file:"):
                return repo_link(name[5:], repo_url)
            if name not in facts:
                raise TemplateError(f"unknown token {name!r}")
            if used is not None:
                used.add(name)
            return fmt(facts[name], how)
        except TemplateError as e:
            raise TemplateError(f"{where}: {{{{{m.group(0)[2:-2]}}}}}: {e}") from None

    return PLACEHOLDER.sub(one, text)


def repo_link(path: str, repo_url: str) -> str:
    """A repo file: a link when the repo URL is known, else code text."""
    if repo_url:
        return f'<a href="{escape(repo_url.rstrip("/"))}/blob/main/{escape(path)}"><code>{escape(path)}</code></a>'
    return f"<code>{escape(path)}</code>"


def render_sections(folder: Path, names: list[str], facts: dict, blocks: dict, repo_url: str = "") -> tuple[dict, set]:
    """{name: filled HTML} for each section file, and the set of facts used."""
    used: set = set()
    out = {}
    for n in names:
        f = folder / f"{n}.html"
        if not f.exists():
            raise TemplateError(f"missing section file {f}")
        out[n] = fill(f.read_text(), facts, blocks, repo_url, used, where=f.name)
    return out, used


# ---- facts ------------------------------------------------------------------------------

def _all_rows(h: pd.DataFrame, mode: str) -> pd.DataFrame:
    return h[(h["mode"] == mode) & (h.site == "All")].set_index("model")


def build_facts(h: pd.DataFrame, p: pd.DataFrame, g: pd.DataFrame, sp: pd.DataFrame, sites: pd.DataFrame,
                tol: float = 0.02, density: pd.DataFrame | None = None) -> dict:
    """Every number the prose may cite, named plainly. Built from the report tables only."""
    f: dict = {}
    per_site = h[h.site != "All"]
    clip_sites = sorted(set(per_site[per_site["mode"] == "clips"].site))
    scape_sites = sorted(set(per_site[per_site["mode"] == "scape"].site))
    f["n_sites"] = len(set(clip_sites) | set(scape_sites))
    f["n_models"] = h.model.nunique()
    f["sites_clips"] = ", ".join(clip_sites)
    f["sites_scape"] = ", ".join(scape_sites)
    f["n_species_pairs"] = int((sp["mode"] == "clips").sum())
    f["n_species_unique"] = int(sp.loc[sp["mode"] == "clips", "species"].nunique())

    # replication against the Perch 2.0 paper
    if not p.empty:
        within = ~p["flag"].astype(str).eq("True")
        f["repl_n_within"] = int(within.sum())
        f["repl_n_total"] = len(p)
        f["repl_sites"] = ", ".join(sorted(p.site.unique()))
        f["repl_n_sites"] = p.site.nunique()
        f["repl_max_below"] = float(-p["delta"].min()) if (p["delta"] < 0).any() else 0.0
    f["repl_tol"] = tol

    # per-model means over sites
    for mode in ("clips", "scape"):
        a = _all_rows(h, mode)
        for m, k in MODEL_KEYS.items():
            if m not in a.index:
                continue
            for col, name in [("cmAP", "cmap"), ("AUROC", "auroc"), ("top1", "top1"),
                              ("rec_cmAP", "rec_cmap"), ("rec_recall90", "rec_recall90")]:
                v = a.loc[m, col] if col in a else np.nan
                if pd.notna(v):
                    f[f"{mode}_{name}_{k}"] = float(v)
        if not a.empty:
            f[f"{mode}_n_sites_mean"] = len(str(a["sites"].iloc[0]).split())

    # hardest site: lowest mean clips cmAP over models
    c = per_site[per_site["mode"] == "clips"]
    if not c.empty:
        by = c.groupby("site")["cmAP"].mean().sort_values()
        hard = by.index[0]
        f["hardest_site"] = hard
        f["hardest_site_name"] = sites.set_index("site").name.get(hard, hard)
        f["hardest_site_cmap_range"] = (float(c[c.site == hard].cmAP.min()), float(c[c.site == hard].cmAP.max()))
        f["hardest_site_species"] = int(c[c.site == hard].n_species.max())
        if density is not None and hard in set(density.site):
            f["hardest_site_species_per_rec"] = float(density.set_index("site").loc[hard, "species_per_recording"])
        f["easiest_site"] = by.index[-1]

    # Perch on POW, clips vs soundscape segment cmAP (the alignment-rule check)
    for mode in ("clips", "scape"):
        r = h[(h["mode"] == mode) & (h.site == "POW") & (h.model == "perch_v2")]
        if not r.empty:
            f[f"perch_pow_{mode}_cmap"] = float(r.cmAP.iloc[0])

    # groupings (clips, all sites)
    cg = g[(g["mode"] == "clips") & (g.scope == "All")]
    for name, label in BIOME_KEYS.items():
        sub = cg[(cg.grouping == "biome") & (cg.group == name)]
        if not sub.empty:
            f[f"biome_{label}_range"] = (float(sub.mean_ap.min()), float(sub.mean_ap.max()))
            f[f"biome_{label}_pairs"] = int(sub.n_pairs.iloc[0])
            f[f"biome_{label}_sites"] = ", ".join(str(sub.sites.iloc[0]).split())
    others = cg[(cg.grouping == "biome") & (cg.group != "Tropical moist broadleaf forest")]
    if not others.empty:
        f["biome_others_range"] = (float(others.mean_ap.min()), float(others.mean_ap.max()))
    for name, label in XC_KEYS.items():
        sub = cg[(cg.grouping == "xc") & (cg.group == name)].set_index("model")
        if sub.empty:
            continue
        f[f"xc_{label}_pairs"] = int(sub.n_pairs.iloc[0])
        f[f"xc_{label}_range"] = (float(sub.mean_ap.min()), float(sub.mean_ap.max()))
        for m, k in MODEL_KEYS.items():
            if m in sub.index:
                f[f"xc_{label}_{k}"] = float(sub.loc[m, "mean_ap"])
    for name, label in PREV_KEYS.items():
        sub = cg[(cg.grouping == "prevalence") & (cg.group == name)]
        if not sub.empty:
            f[f"prev_{label}_range"] = (float(sub.mean_ap.min()), float(sub.mean_ap.max()))
            f[f"prev_{label}_pairs"] = int(sub.n_pairs.iloc[0])
    return f

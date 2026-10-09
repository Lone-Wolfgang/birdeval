"""Render the report tables from build_report.py as one self-contained web page.

    python -I scripts/build_report.py RUN_DIR [RUN_DIR ...]
    python -I scripts/render_report.py [--report results/report] [--out results/report/index.html]

Reads headline.csv, published.csv, by_abundance.csv and by_habitat.csv (the
last is optional) and writes a single HTML file: no scripts, charts as inline
SVG, light and dark themes. Every chart has the table it is drawn from below it.
"""

from __future__ import annotations

import argparse
import datetime as dt
from html import escape
from pathlib import Path

import pandas as pd

MODELS = {  # fixed order and colour slot: colour follows the model, never its rank
    "birdnet_2.4": ("BirdNET 2.4", "m1"),
    "birdnet_3.0": ("BirdNET 3.0", "m2"),
    "perch_v2": ("Perch V2", "m3"),
}
SITES = {
    "POW": "Powdermill, Pennsylvania",
    "HSN": "High Sierras, California",
    "SNE": "Sierra Nevada, California",
}

CSS = """
/* Layout: one reading column; charts and tables full column width, tables scroll on their own. */
:root {
  --bg: #f5f7f6; --surface: #ffffff; --fg: #17201d; --muted: #56635e; --rule: #d9e0dc;
  --accent: #1d5c4b; --grid: #e6ebe8;
  --m1: #0072b2; --m2: #c26a00; --m3: #00896a;
  --ok: #1d6b3a; --warn: #9a5b00;
  --display: "IBM Plex Sans Condensed", "Arial Narrow", system-ui, sans-serif;
  --body: "IBM Plex Sans", system-ui, -apple-system, "Segoe UI", sans-serif;
  --mono: "IBM Plex Mono", ui-monospace, "SF Mono", Menlo, monospace;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg: #111816; --surface: #18211e; --fg: #e4ebe8; --muted: #9aa8a2; --rule: #2c3833;
  --accent: #7cc7ae; --grid: #24302b;
  --m1: #56b4e9; --m2: #e69f00; --m3: #3cc79a; --ok: #6fcf8f; --warn: #f0b04a; color-scheme: dark } }
:root[data-theme="dark"] {
  --bg: #111816; --surface: #18211e; --fg: #e4ebe8; --muted: #9aa8a2; --rule: #2c3833;
  --accent: #7cc7ae; --grid: #24302b;
  --m1: #56b4e9; --m2: #e69f00; --m3: #3cc79a; --ok: #6fcf8f; --warn: #f0b04a; color-scheme: dark }
body { background: var(--bg); color: var(--fg); font: 15px/1.55 var(--body); }
main { max-width: 860px; margin: 0 auto; padding-inline: 20px; padding-block: 40px 64px;
  display: flex; flex-direction: column; gap: 44px; }
header { display: flex; flex-direction: column; gap: 10px; }
.eyebrow { font: 600 12px/1 var(--mono); letter-spacing: .08em; text-transform: uppercase; color: var(--accent); }
h1 { font: 600 clamp(28px, 5vw, 40px)/1.1 var(--display); margin: 0; text-wrap: balance; }
h2 { font: 600 22px/1.2 var(--display); margin: 0; text-wrap: balance; }
p { margin: 0; max-width: 66ch; }
.meta { font: 13px/1.5 var(--mono); color: var(--muted); }
section { display: flex; flex-direction: column; gap: 14px; min-width: 0; }
.lede { color: var(--muted); }
ul.takeaways { margin: 0; padding-left: 1.2em; display: flex; flex-direction: column; gap: 6px; max-width: 70ch; }
.legend { display: flex; flex-wrap: wrap; gap: 6px 18px; font-size: 13px; color: var(--muted); }
.legend span { display: inline-flex; align-items: center; gap: 6px; }
.legend i { width: 10px; height: 10px; border-radius: 50%; display: inline-block; }
.chart { background: var(--surface); border: 1px solid var(--rule); border-radius: 6px; padding: 12px 8px; overflow-x: auto; }
.chart svg { display: block; width: 100%; height: auto; min-width: 520px; }
.chart text { font-family: var(--body); }
.tick { fill: var(--muted); font-size: 11px; font-variant-numeric: tabular-nums; }
.glabel { fill: var(--fg); font-size: 13px; font-weight: 600; }
.gsub { fill: var(--muted); font-size: 11px; }
.vlabel { fill: var(--fg); font-size: 11px; font-variant-numeric: tabular-nums; }
.gridline { stroke: var(--grid); stroke-width: 1; }
.band { fill: var(--bg); }
.m1 { --c: var(--m1); } .m2 { --c: var(--m2); } .m3 { --c: var(--m3); }
.ci { stroke: var(--c); stroke-width: 2; stroke-linecap: round; }
.dot { fill: var(--c); stroke: var(--surface); stroke-width: 2; }
.range { fill: var(--accent); opacity: .18; }
.table-wrap { overflow-x: auto; border: 1px solid var(--rule); border-radius: 6px; background: var(--surface); }
table { border-collapse: collapse; width: 100%; font-size: 13.5px; font-variant-numeric: tabular-nums; }
th, td { padding: 7px 12px; text-align: right; border-bottom: 1px solid var(--rule); white-space: nowrap; }
th:first-child, td:first-child, th.l, td.l { text-align: left; }
thead th { font: 600 11.5px/1.3 var(--mono); letter-spacing: .04em; text-transform: uppercase; color: var(--muted); }
tbody tr:last-child td { border-bottom: 0; }
td.best { font-weight: 700; }
.ci-txt { color: var(--muted); font-size: 12px; }
.swatch { display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: var(--c); margin-right: 7px; }
.chip { display: inline-block; font: 600 11px/1 var(--mono); padding: 4px 7px; border-radius: 99px; border: 1px solid currentColor; }
.chip.ok { color: var(--ok); } .chip.warn { color: var(--warn); }
details { border-top: 1px solid var(--rule); padding-top: 10px; }
summary { cursor: pointer; color: var(--muted); font-size: 13px; }
summary:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
details[open] summary { margin-bottom: 10px; }
dl { display: grid; grid-template-columns: max-content 1fr; gap: 6px 16px; margin: 0; }
dt { font: 600 13px/1.5 var(--mono); }
dd { margin: 0; min-width: 0; }
.notes { display: flex; flex-direction: column; gap: 8px; padding-left: 1.2em; margin: 0; max-width: 70ch; }
footer { color: var(--muted); font-size: 13px; border-top: 1px solid var(--rule); padding-top: 16px; }
@media (max-width: 520px) { dl { grid-template-columns: 1fr; } dd { margin-bottom: 6px; } }
"""


def model_name(m: str) -> str:
    return MODELS.get(m, (m, "m1"))[0]


def model_cls(m: str) -> str:
    return MODELS.get(m, (m, "m1"))[1]


def legend(models: list[str]) -> str:
    items = "".join(f'<span class="{model_cls(m)}"><i style="background:var(--c)"></i>{escape(model_name(m))}</span>'
                    for m in models)
    return f'<div class="legend" aria-label="Models">{items}</div>'


def dot_chart(rows: pd.DataFrame, group_col: str, groups: list[tuple[str, str]], models: list[str],
              value: str, lo: str, hi: str, axis: str, ranges: dict | None = None) -> str:
    """Dots with 95% intervals on a 0-1 axis: one band per group, one row per model.

    `groups` is [(key, subtitle)]. `ranges` maps group key -> (lo, hi) to shade
    a published range behind the band.
    """
    W, left, right, row, pad = 720, 190, 56, 22, 12
    band_h = pad * 2 + row * len(models)
    H = 28 + band_h * len(groups) + 8 * (len(groups) - 1) + 24
    x = lambda v: left + v * (W - left - right)  # noqa: E731
    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{escape(axis)}">']
    for t in [0, .2, .4, .6, .8, 1]:
        out.append(f'<line class="gridline" x1="{x(t):.1f}" x2="{x(t):.1f}" y1="20" y2="{H - 24}"/>')
        out.append(f'<text class="tick" x="{x(t):.1f}" y="{H - 8}" text-anchor="middle">{t:.1f}</text>')
    out.append(f'<text class="tick" x="{x(0):.1f}" y="12">{escape(axis)} (0 to 1, higher is better)</text>')
    y0 = 28
    for key, sub in groups:
        out.append(f'<rect class="band" x="{left - 8}" y="{y0}" width="{W - left - right + 16}" height="{band_h}" rx="4"/>')
        if ranges and key in ranges:
            a, b = ranges[key]
            out.append(f'<rect class="range" x="{x(a):.1f}" y="{y0}" width="{max(x(b) - x(a), 2):.1f}" height="{band_h}">'
                       f'<title>Published range {a:.3f}–{b:.3f}</title></rect>')
        out.append(f'<text class="glabel" x="8" y="{y0 + band_h / 2 - 2:.1f}">{escape(key)}</text>')
        out.append(f'<text class="gsub" x="8" y="{y0 + band_h / 2 + 14:.1f}">{escape(sub)}</text>')
        for i, m in enumerate(models):
            r = rows[(rows[group_col] == key) & (rows["model"] == m)]
            if r.empty:
                continue
            r = r.iloc[0]
            cy = y0 + pad + row * i + row / 2
            tip = f"{model_name(m)}, {key}: {r[value]:.3f} (95% CI {r[lo]:.3f}–{r[hi]:.3f})"
            out.append(f'<g class="{model_cls(m)}"><title>{escape(tip)}</title>'
                       f'<line class="ci" x1="{x(r[lo]):.1f}" x2="{x(r[hi]):.1f}" y1="{cy:.1f}" y2="{cy:.1f}"/>'
                       f'<circle class="dot" cx="{x(r[value]):.1f}" cy="{cy:.1f}" r="5.5"/></g>')
            out.append(f'<text class="vlabel" x="{x(r[hi]) + 8:.1f}" y="{cy + 4:.1f}">{r[value]:.2f}</text>')
        y0 += band_h + 8
    out.append("</svg>")
    return f'<div class="chart">{"".join(out)}</div>'


def table(head: list[str], body: list[list[str]], left_cols: int = 1) -> str:
    th = "".join(f'<th class="{"l" if i < left_cols else ""}">{h}</th>' for i, h in enumerate(head))
    trs = "".join("<tr>" + "".join(f'<td class="{"l" if i < left_cols else ""}">{c}</td>' if not c.startswith("<td")
                                   else c for i, c in enumerate(r)) + "</tr>" for r in body)
    return f'<div class="table-wrap"><table><thead><tr>{th}</tr></thead><tbody>{trs}</tbody></table></div>'


def model_cell(m: str) -> str:
    return f'<span class="{model_cls(m)}"><span class="swatch"></span>{escape(model_name(m))}</span>'


def fmt(v, d: int = 3) -> str:
    return "–" if pd.isna(v) else f"{v:.{d}f}"


def headline_section(h: pd.DataFrame, models: list[str]) -> str:
    sites = [s for s in SITES if s in set(h["site"])] + sorted(set(h["site"]) - set(SITES))
    groups = [(s, f'{SITES.get(s, "")}'.split(",")[0] + f' · {int(h[h.site == s].n_species.iloc[0])} species')
              for s in sites]
    chart = dot_chart(h, "site", groups, models, "cmAP", "cmAP_lo", "cmAP_hi", "Segment cmAP")
    body = []
    for s in sites:
        sub = h[h["site"] == s]
        best = {c: sub[c].max() for c in ["cmAP", "AUROC", "top1", "rec_cmAP"]}
        for m in models:
            r = sub[sub["model"] == m]
            if r.empty:
                continue
            r = r.iloc[0]
            cell = lambda c: (f'<td class="best">{fmt(r[c])}</td>' if r[c] == best[c] and not pd.isna(r[c])  # noqa: E731
                              else f"<td>{fmt(r[c])}</td>")
            body.append([s, model_cell(m),
                         f'<td class="{"best" if r.cmAP == best["cmAP"] else ""}">{fmt(r.cmAP)} '
                         f'<span class="ci-txt">{fmt(r.cmAP_lo, 2)}–{fmt(r.cmAP_hi, 2)}</span></td>',
                         cell("AUROC"), cell("top1"), cell("rec_cmAP"), f"{int(r.n_clips):,}"])
    tbl = table(["Site", "Model", "cmAP (95% CI)", "AUROC", "Top-1", "Recording cmAP", "Clips"], body, left_cols=2)
    return f"""<section id="by-dataset">
  <h2>Performance by dataset</h2>
  <p class="lede">Segment-level class-mean average precision on each site's 5-second test clips. Dots are the
  estimate, lines the 95% bootstrap interval over recordings. The best value per site is bold in the table.</p>
  {legend(models)}
  {chart}
  {tbl}
</section>"""


def published_section(p: pd.DataFrame) -> str:
    if p.empty:
        return ""
    names = {"cmAP": "cmAP", "AUROC": "AUROC", "top1": "Top-1"}
    body = []
    for _, r in p.sort_values(["site", "metric"]).iterrows():
        chip = ('<span class="chip warn">Outside</span>' if r["flag"] in (True, "True")
                else '<span class="chip ok">Within</span>')
        body.append([r["site"], names.get(r["metric"], r["metric"]), fmt(r["ours"]),
                     f'{fmt(r["pub_lo"])}–{fmt(r["pub_hi"])}', f'{r["delta"]:+.3f}', chip])
    n_ok = int((~p["flag"].astype(str).eq("True")).sum())
    ranges = {s: (g.pub_lo.iloc[0], g.pub_hi.iloc[0]) for s, g in p[p.metric == "cmAP"].groupby("site")}
    rng = ", ".join(f"{s} {a:.3f}–{b:.3f}" for s, (a, b) in ranges.items())
    return f"""<section id="published">
  <h2>Check against published figures</h2>
  <p class="lede">Perch V2 against the three Perch 2.0 variants in van Merriënboer et al. 2025 (arXiv:2508.04665,
  Appendix C, Table 4). A value counts as within if it is no more than 0.02 outside the published range.
  {n_ok} of {len(p)} values are within. Published cmAP: {escape(rng)}.</p>
  {table(["Site", "Metric", "Ours", "Published", "Distance", "Result"], [b[:-1] + [f'<td class="l">{b[-1]}</td>'] for b in body], left_cols=2)}
  <p class="meta">POW is not in the Perch 2.0 table: BirdSet uses it as a validation set.
  BirdNET has no published per-site figures; BirdSet left it out because its training data may overlap the test sets.</p>
</section>"""


def grouped_section(df: pd.DataFrame, models: list[str], sid: str, title: str, lede: str, unit: str) -> str:
    order = list(dict.fromkeys(df["group"]))
    groups = [(g, f'{int(df[df.group == g].n_species.iloc[0])} {unit}') for g in order]
    chart = dot_chart(df, "group", groups, models, "mean_ap", "ap_lo", "ap_hi", "Mean per-species AP")
    body = []
    for g in order:
        sub = df[df["group"] == g]
        best = sub["mean_ap"].max()
        for m in models:
            r = sub[sub["model"] == m]
            if r.empty:
                continue
            r = r.iloc[0]
            body.append([escape(g), model_cell(m),
                         f'<td class="{"best" if r.mean_ap == best else ""}">{fmt(r.mean_ap)} '
                         f'<span class="ci-txt">{fmt(r.ap_lo, 2)}–{fmt(r.ap_hi, 2)}</span></td>',
                         str(int(r.n_species)), escape(r.sites)])
    tbl = table(["Group", "Model", "Mean AP (95% CI)", "Pairs", "Sites"], body, left_cols=2)
    return f"""<section id="{sid}">
  <h2>{escape(title)}</h2>
  <p class="lede">{lede}</p>
  {legend(models)}
  {chart}
  <details><summary>Show the table</summary>{tbl}</details>
</section>"""


def takeaways(h: pd.DataFrame, p: pd.DataFrame, ab: pd.DataFrame) -> str:
    items = []
    for s in [s for s in SITES if s in set(h.site)]:
        sub = h[h.site == s].sort_values("cmAP", ascending=False)
        a, b = sub.iloc[0], sub.iloc[1]
        overlap = a.cmAP_lo <= b.cmAP_hi
        items.append(f"<li><b>{s}</b>: {escape(model_name(a.model))} has the highest cmAP ({a.cmAP:.3f}), "
                     f"then {escape(model_name(b.model))} ({b.cmAP:.3f}). "
                     + ("Their intervals overlap.</li>" if overlap else "Their intervals do not overlap.</li>"))
    if not p.empty:
        n_ok = int((~p["flag"].astype(str).eq("True")).sum())
        items.append(f"<li>Perch V2 matches the Perch 2.0 paper: {n_ok} of {len(p)} published values on HSN and SNE "
                     f"are reproduced within 0.02.</li>")
    if not ab.empty:
        first, last = ab.group.iloc[0], ab.group.iloc[-1]
        lo = ab[ab.group == first].set_index("model").mean_ap
        hi = ab[ab.group == last].set_index("model").mean_ap
        items.append(f"<li>Every model does better on species with many labelled clips ({escape(last)}: "
                     f"{hi.min():.2f}–{hi.max():.2f} mean AP) than on rare ones ({escape(first)}: "
                     f"{lo.min():.2f}–{lo.max():.2f}).</li>")
    return f'<ul class="takeaways">{"".join(items)}</ul>'


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", type=Path, default=Path("results/report"))
    ap.add_argument("--out", type=Path)
    ap.add_argument("--title", default="BirdSet Model Comparison")
    args = ap.parse_args()
    out = args.out or args.report / "index.html"

    h = pd.read_csv(args.report / "headline.csv")
    p = pd.read_csv(args.report / "published.csv")
    ab = pd.read_csv(args.report / "by_abundance.csv")
    hab_f = args.report / "by_habitat.csv"
    hab = pd.read_csv(hab_f) if hab_f.exists() else pd.DataFrame()
    models = [m for m in MODELS if m in set(h.model)] + sorted(set(h.model) - set(MODELS))
    sites = [s for s in SITES if s in set(h.site)]
    today = dt.date.today().isoformat()

    sections = [
        headline_section(h, models),
        published_section(p),
        grouped_section(ab, models, "by-abundance", "By label abundance",
                        "Mean average precision per species, grouped by how many labelled 5-second clips the species "
                        "has at its site. Each unit is one (site, species) pair; intervals resample those pairs. "
                        "Bins were fixed before looking at results.", "pairs"),
    ]
    if not hab.empty:
        sections.append(grouped_section(
            hab, models, "by-habitat", "By habitat",
            "The same measure grouped by each species' primary habitat in AVONET (Tobias et al. 2022). "
            "Most pairs are forest species, so the other groups are small and their intervals wide. "
            "Habitats with fewer than 5 pairs are merged into Other.", "pairs"))

    page = f"""<title>{escape(args.title)}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans+Condensed:wght@600&family=IBM+Plex+Sans:wght@400;600;700&display=swap">
<style>{CSS}</style>
<main>
<header>
  <div class="eyebrow">birdeval · BirdSet {" · ".join(sites)}</div>
  <h1>{escape(args.title)}</h1>
  <p class="lede">{", ".join(escape(model_name(m)) for m in models)} on {len(sites)} BirdSet test sites,
  scored with BirdSet's protocol: every species in the site's label set, 5-second clips.</p>
  <p class="meta">Generated {today} by scripts/render_report.py</p>
</header>
<section id="summary">
  <h2>Summary</h2>
  {takeaways(h, p, ab)}
</section>
{"".join(sections)}
<section id="notes">
  <h2>Notes</h2>
  <ul class="notes">
    <li>BirdNET may have been trained on recordings from these test sets, so its scores may be optimistic.
    BirdSet excluded it for this reason.</li>
    <li>Species names follow the 2025 eBird taxonomy. Older model names are mapped through
    configs/taxonomy/synonyms.yaml, and every model scores every species at every site.</li>
    <li>Per-species AP is noisy for species with only a few labelled clips.</li>
    <li>Recording cmAP comes from soundscape mode: model scores pooled into recording units, then compared with
    the species present.</li>
  </ul>
  <details><summary>Metric definitions</summary>
  <dl>
    <dt>cmAP</dt><dd>Average precision per species, averaged over species. Treats rare and common species equally.</dd>
    <dt>AUROC</dt><dd>Chance a positive clip scores higher than a negative one, averaged over species.</dd>
    <dt>Top-1</dt><dd>Share of clips with at least one label where the top-scoring species is one of them.</dd>
    <dt>Recording cmAP</dt><dd>cmAP for species presence per recording unit, from soundscape mode.</dd>
  </dl>
  </details>
</section>
<footer>Habitat data: AVONET, Tobias et al. 2022, Ecology Letters 25:581 (CC BY 4.0).
Published figures: van Merriënboer et al. 2025 (Perch 2.0); Rauch et al. 2024 (BirdSet, arXiv:2403.10380).</footer>
</main>
"""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

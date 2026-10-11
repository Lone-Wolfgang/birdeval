"""Render the report tables from build_report.py as one self-contained web page.

    python -I scripts/build_report.py RUN_DIR [RUN_DIR ...]
    python -I scripts/render_report.py [--report results/report] [--out results/report/index.html]

Reads headline.csv, published.csv, species.csv and groups.csv, plus
configs/reference/sites.csv, and writes a single HTML file. The summary and the
published check are static; the explorer (mode, dataset and view; sortable
tables; dot charts) is a small inline script over the same data, embedded as
JSON. No external scripts; light and dark themes.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from html import escape
from pathlib import Path

import pandas as pd

MODELS = {  # fixed order and colour slot: colour follows the model, never its rank
    "birdnet_2.4": ("BirdNET 2.4", "m1"),
    "birdnet_3.0": ("BirdNET 3.0", "m2"),
    "perch_v2": ("Perch V2", "m3"),
}

CSS = """
/* Layout: one reading column; charts and tables full column width, tables scroll on their own. */
:root {
  --bg: #f5f7f6; --surface: #ffffff; --fg: #17201d; --muted: #56635e; --rule: #d9e0dc;
  --accent: #1d5c4b; --accent-fg: #ffffff; --grid: #e6ebe8; --hover: #eef3f0;
  --m1: #0072b2; --m2: #c26a00; --m3: #00896a;
  --ok: #1d6b3a; --warn: #9a5b00;
  --display: "IBM Plex Sans Condensed", "Arial Narrow", system-ui, sans-serif;
  --body: "IBM Plex Sans", system-ui, -apple-system, "Segoe UI", sans-serif;
  --mono: "IBM Plex Mono", ui-monospace, "SF Mono", Menlo, monospace;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg: #111816; --surface: #18211e; --fg: #e4ebe8; --muted: #9aa8a2; --rule: #2c3833;
  --accent: #7cc7ae; --accent-fg: #0d1513; --grid: #24302b; --hover: #1f2a26;
  --m1: #56b4e9; --m2: #e69f00; --m3: #3cc79a; --ok: #6fcf8f; --warn: #f0b04a; color-scheme: dark } }
:root[data-theme="dark"] {
  --bg: #111816; --surface: #18211e; --fg: #e4ebe8; --muted: #9aa8a2; --rule: #2c3833;
  --accent: #7cc7ae; --accent-fg: #0d1513; --grid: #24302b; --hover: #1f2a26;
  --m1: #56b4e9; --m2: #e69f00; --m3: #3cc79a; --ok: #6fcf8f; --warn: #f0b04a; color-scheme: dark }
body { background: var(--bg); color: var(--fg); font: 15px/1.55 var(--body); }
main { max-width: 960px; margin: 0 auto; padding-inline: 20px; padding-block: 40px 64px;
  display: flex; flex-direction: column; gap: 44px; }
@media (max-width: 520px) { main { padding-inline: 16px; } }
header { display: flex; flex-direction: column; gap: 10px; }
.eyebrow { font: 600 12px/1 var(--mono); letter-spacing: .08em; text-transform: uppercase; color: var(--accent); }
h1 { font: 600 clamp(28px, 5vw, 40px)/1.1 var(--display); margin: 0; text-wrap: balance; }
h2 { font: 600 22px/1.2 var(--display); margin: 0; text-wrap: balance; }
h3 { font: 600 17px/1.3 var(--display); margin: 0; }
p { margin: 0; max-width: 70ch; }
.meta { font: 13px/1.5 var(--mono); color: var(--muted); }
section { display: flex; flex-direction: column; gap: 14px; min-width: 0; }
#view { display: flex; flex-direction: column; gap: 14px; min-width: 0; }
.lede { color: var(--muted); }
ul.takeaways { margin: 0; padding-left: 1.2em; display: flex; flex-direction: column; gap: 6px; max-width: 74ch; }
.legend { display: flex; flex-wrap: wrap; gap: 6px 18px; font-size: 13px; color: var(--muted); }
.legend span { display: inline-flex; align-items: center; gap: 6px; }
.legend i { width: 10px; height: 10px; border-radius: 50%; display: inline-block; background: var(--c); }

/* Explorer controls */
.controls { display: flex; flex-wrap: wrap; gap: 12px 24px; align-items: flex-end;
  padding: 14px 16px; background: var(--surface); border: 1px solid var(--rule); border-radius: 6px; }
.ctl { display: flex; flex-direction: column; gap: 6px; min-width: 0; }
.ctl-label { font: 600 11px/1 var(--mono); letter-spacing: .06em; text-transform: uppercase; color: var(--muted); }
.seg { display: flex; flex-wrap: wrap; border: 1px solid var(--rule); border-radius: 6px; overflow: hidden; }
.seg button { font: 500 13.5px/1 var(--body); color: var(--fg); background: var(--surface); border: 0;
  border-right: 1px solid var(--rule); padding: 8px 12px; cursor: pointer; }
.seg button:last-child { border-right: 0; }
.seg button:hover:not(:disabled) { background: var(--hover); }
.seg button[aria-pressed="true"] { background: var(--accent); color: var(--accent-fg); }
.seg button:disabled { color: var(--muted); opacity: .55; cursor: not-allowed; }
select, input[type="search"] { font: 14px/1.2 var(--body); color: var(--fg); background: var(--surface);
  border: 1px solid var(--rule); border-radius: 6px; padding: 7px 10px; max-width: 100%; }
input[type="search"] { width: 16em; }
.check { display: inline-flex; gap: 8px; align-items: center; font-size: 14px; cursor: pointer; }
button:focus-visible, select:focus-visible, input:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.subcontrols { display: flex; flex-wrap: wrap; gap: 12px 20px; align-items: flex-end; }

.chart { background: var(--surface); border: 1px solid var(--rule); border-radius: 6px; padding: 12px 8px; overflow-x: auto; }
.chart svg { display: block; width: 100%; height: auto; min-width: 560px; }
.chart text { font-family: var(--body); }
.tick { fill: var(--muted); font-size: 11px; font-variant-numeric: tabular-nums; }
.glabel { fill: var(--fg); font-size: 12.5px; font-weight: 600; }
.gsub { fill: var(--muted); font-size: 11px; }
.vlabel { fill: var(--fg); font-size: 11px; font-variant-numeric: tabular-nums; }
.gridline { stroke: var(--grid); stroke-width: 1; }
.band { fill: var(--bg); }
.m1 { --c: var(--m1); } .m2 { --c: var(--m2); } .m3 { --c: var(--m3); }
.ci { stroke: var(--c); stroke-width: 2; stroke-linecap: round; }
.dot { fill: var(--c); stroke: var(--surface); stroke-width: 2; }
.range { fill: var(--accent); opacity: .18; }

.table-wrap { overflow-x: auto; border: 1px solid var(--rule); border-radius: 6px; background: var(--surface); }
.table-wrap.tall { max-height: 70vh; overflow-y: auto; }
table { border-collapse: collapse; width: 100%; font-size: 13.5px; font-variant-numeric: tabular-nums; }
th, td { padding: 7px 12px; text-align: right; border-bottom: 1px solid var(--rule); white-space: nowrap; }
th.l, td.l { text-align: left; }
thead th { font: 600 11.5px/1.3 var(--mono); letter-spacing: .04em; text-transform: uppercase; color: var(--muted);
  background: var(--surface); position: sticky; top: 0; z-index: 1; }
th button { font: inherit; letter-spacing: inherit; text-transform: inherit; color: inherit; background: none;
  border: 0; padding: 0; cursor: pointer; display: inline-flex; gap: 4px; align-items: center; }
th button:hover { color: var(--fg); }
th[aria-sort] button { color: var(--fg); }
th .arrow { font-size: 10px; width: 1ch; }
tbody tr:last-child td { border-bottom: 0; }
tbody tr:hover td { background: var(--hover); }
tr.total td { border-top: 2px solid var(--rule); font-weight: 600; }
td.best { font-weight: 700; }
td.dim { color: var(--muted); }
.ci-txt { color: var(--muted); font-size: 12px; font-weight: 400; }
.swatch { display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: var(--c); margin-right: 7px; }
.sp { font-style: italic; }
.fam { color: var(--muted); font-size: 12px; }
.chip { display: inline-block; font: 600 11px/1 var(--mono); padding: 4px 7px; border-radius: 99px; border: 1px solid currentColor; }
.chip.ok { color: var(--ok); } .chip.warn { color: var(--warn); }
.empty { padding: 18px; color: var(--muted); background: var(--surface); border: 1px dashed var(--rule); border-radius: 6px; }

details { border-top: 1px solid var(--rule); padding-top: 10px; }
summary { cursor: pointer; color: var(--muted); font-size: 13px; }
summary:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
details[open] summary { margin-bottom: 10px; }
dl { display: grid; grid-template-columns: max-content 1fr; gap: 6px 16px; margin: 0; }
dt { font: 600 13px/1.5 var(--mono); }
dd { margin: 0; min-width: 0; }
.notes { display: flex; flex-direction: column; gap: 8px; padding-left: 1.2em; margin: 0; max-width: 74ch; }
footer { color: var(--muted); font-size: 13px; border-top: 1px solid var(--rule); padding-top: 16px; }
@media (max-width: 520px) { dl { grid-template-columns: 1fr; } dd { margin-bottom: 6px; } input[type="search"] { width: 100%; } }
"""

JS = r"""
const D = JSON.parse(document.getElementById("report-data").textContent);
const MODELS = D.models, SITES = D.sites;
const SITE_ORDER = Object.fromEntries(SITES.map((s, i) => [s.site, i]));
const VIEWS = ["overview", "habitat", "biome", "prevalence", "species"];
const RARE = 10;
const KEY = "birdeval-report-v2";

const state = Object.assign(
  {mode: "clips", site: "All", view: "overview", metric: "ap", dA: MODELS[0].id,
   dB: MODELS[MODELS.length - 1].id, q: "", hideRare: true, sort: {}},
  load());
const tok = location.hash.slice(1);
if (SITE_ORDER[tok] !== undefined) state.site = tok;
if (VIEWS.includes(tok)) state.view = tok;
if (state.site !== "All" && SITE_ORDER[state.site] === undefined) state.site = "All";

function load() { try { return JSON.parse(localStorage.getItem(KEY)) || {}; } catch (e) { return {}; } }
function save() {
  try { const {q, ...rest} = state; localStorage.setItem(KEY, JSON.stringify(rest)); } catch (e) { /* storage blocked */ }
}

const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c]));
const num = v => v !== null && v !== undefined && !Number.isNaN(v);
const f3 = v => num(v) ? v.toFixed(3) : "–";
const f2 = v => num(v) ? v.toFixed(2) : "–";
const sgn = v => num(v) ? (v > 0 ? "+" : v < 0 ? "−" : "") + Math.abs(v).toFixed(3) : "–";
const cap = s => s.charAt(0).toUpperCase() + s.slice(1);
const siteName = s => (SITES[SITE_ORDER[s]] || {}).name || s;
const modelCell = m => `<span class="${m.cls}"><span class="swatch"></span>${esc(m.name)}</span>`;
const legend = () => `<div class="legend" aria-label="Models">${MODELS.map(m =>
  `<span class="${m.cls}"><i></i>${esc(m.name)}</span>`).join("")}</div>`;

// ---- dot chart: one band per group, one row per model, 0-1 axis -------------
function dotChart(groups, get, axis) {
  // groups: [{key, label, sub}], get(group, model) -> {v, lo, hi} | null
  const W = 760, left = 250, right = 56, row = 22, pad = 12;
  const bandH = pad * 2 + row * MODELS.length;
  const H = 28 + bandH * groups.length + 8 * (groups.length - 1) + 24;
  const x = v => left + v * (W - left - right);
  const out = [`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(axis)}">`];
  for (const t of [0, .2, .4, .6, .8, 1]) {
    out.push(`<line class="gridline" x1="${x(t)}" x2="${x(t)}" y1="20" y2="${H - 24}"/>`,
             `<text class="tick" x="${x(t)}" y="${H - 8}" text-anchor="middle">${t.toFixed(1)}</text>`);
  }
  out.push(`<text class="tick" x="${x(0)}" y="12">${esc(axis)} (0 to 1, higher is better)</text>`);
  let y0 = 28;
  for (const g of groups) {
    out.push(`<rect class="band" x="${left - 8}" y="${y0}" width="${W - left - right + 16}" height="${bandH}" rx="4"/>`);
    out.push(`<text class="glabel" x="8" y="${y0 + bandH / 2 - 2}"><title>${esc(g.label)}</title>${esc(g.label.length > 34 ? g.label.slice(0, 33) + "…" : g.label)}</text>`);
    out.push(`<text class="gsub" x="8" y="${y0 + bandH / 2 + 14}">${esc(g.sub)}</text>`);
    MODELS.forEach((m, i) => {
      const r = get(g, m);
      if (!r || !num(r.v)) return;
      const cy = y0 + pad + row * i + row / 2;
      const hasCI = num(r.lo) && num(r.hi);
      const tip = `${m.name}, ${g.label}: ${f3(r.v)}` + (hasCI ? ` (95% CI ${f3(r.lo)}–${f3(r.hi)})` : "");
      out.push(`<g class="${m.cls}"><title>${esc(tip)}</title>` +
        (hasCI ? `<line class="ci" x1="${x(r.lo)}" x2="${x(r.hi)}" y1="${cy}" y2="${cy}"/>` : "") +
        `<circle cx="${x(r.v)}" cy="${cy}" r="11" fill="transparent"/>` +
        `<circle class="dot" cx="${x(r.v)}" cy="${cy}" r="5.5"/></g>`);
      out.push(`<text class="vlabel" x="${x(hasCI ? r.hi : r.v) + 9}" y="${cy + 4}">${f2(r.v)}</text>`);
    });
    y0 += bandH + 8;
  }
  out.push("</svg>");
  return `<div class="chart">${out.join("")}</div>`;
}

// ---- sortable table ------------------------------------------------------------
// cols: [{k, label, left, cell(row) -> html, sort(row) -> value, title}]
function table(id, cols, rows, opts = {}) {
  const s = state.sort[id];
  if (s) {
    const col = cols.find(c => c.k === s.k);
    if (col) {
      const key = col.sort || (r => r[col.k]);
      rows = rows.slice().sort((a, b) => {
        const va = key(a), vb = key(b);
        const na = !num(va) && typeof va !== "string", nb = !num(vb) && typeof vb !== "string";
        if (na || nb) return na - nb;  // missing values last, whatever the direction
        const c = typeof va === "string" ? va.localeCompare(vb) : va - vb;
        return s.dir === "asc" ? c : -c;
      });
    }
  }
  const th = cols.map(c => {
    const on = s && s.k === c.k;
    const aria = on ? ` aria-sort="${s.dir === "asc" ? "ascending" : "descending"}"` : "";
    const arrow = on ? (s.dir === "asc" ? "▲" : "▼") : "";
    return `<th class="${c.left ? "l" : ""}"${aria}${c.title ? ` title="${esc(c.title)}"` : ""}>` +
      `<button data-table="${id}" data-sort="${c.k}">${c.label}<span class="arrow">${arrow}</span></button></th>`;
  }).join("");
  const body = rows.map(r => `<tr class="${r._cls || ""}">` + cols.map(c => {
    const v = c.cell ? c.cell(r) : esc(r[c.k]);
    return v.startsWith("<td") ? v : `<td class="${c.left ? "l" : ""}">${v}</td>`;
  }).join("") + "</tr>").join("");
  return `<div class="table-wrap${opts.tall ? " tall" : ""}"><table><thead><tr>${th}</tr></thead><tbody>${body}</tbody></table></div>`;
}

function onSort(id, k, numeric) {
  const s = state.sort[id];
  state.sort[id] = s && s.k === k ? {k, dir: s.dir === "asc" ? "desc" : "asc"} : {k, dir: numeric ? "desc" : "asc"};
}

// ---- overview --------------------------------------------------------------------
function overview() {
  const scape = state.mode === "scape";
  let rows = D.headline.filter(r => r.mode === state.mode && (state.site === "All" || r.site === state.site));
  if (!rows.length) return `<div class="empty">No ${scape ? "soundscape" : "clips"} results for ${esc(state.site)} yet.</div>`;
  const allRow = rows.find(r => r.site === "All");
  const siteRows = rows.filter(r => r.site !== "All");
  const sites = [...new Set(siteRows.map(r => r.site))].sort((a, b) => SITE_ORDER[a] - SITE_ORDER[b]);
  const metric = scape ? "rec_cmAP" : "cmAP";
  const groups = sites.map(s => {
    const r = siteRows.find(r => r.site === s);
    return {key: s, label: `${s} · ${siteName(s)}`, sub: `${r.n_species} species`};
  });
  if (allRow) groups.push({key: "All", label: "All sites (mean)", sub: `${allRow.sites.split(" ").length} sites`});
  const chart = dotChart(groups, (g, m) => {
    const r = rows.find(r => r.site === g.key && r.model === m.id);
    return r && {v: r[metric], lo: scape ? null : r.cmAP_lo, hi: scape ? null : r.cmAP_hi};
  }, scape ? "Recording-level cmAP" : "Segment cmAP");

  const mcols = scape
    ? [["rec_cmAP", "Rec. cmAP"], ["rec_recall90", "Recall @90%P"], ["cmAP", "Seg. cmAP"], ["AUROC", "AUROC"], ["top1", "Top-1"]]
    : [["cmAP", "cmAP"], ["AUROC", "AUROC"], ["top1", "Top-1"], ["rec_recall90", "Recall @90%P"]];
  const best = {};
  for (const r of rows) for (const [k] of mcols) {
    const id = r.site + "|" + k;
    if (num(r[k]) && !(r[k] <= best[id])) best[id] = r[k];
  }
  const ordered = [...siteRows.sort((a, b) => SITE_ORDER[a.site] - SITE_ORDER[b.site] ||
    MODELS.findIndex(m => m.id === a.model) - MODELS.findIndex(m => m.id === b.model))];
  ordered.forEach((r, i) => r._i = i);
  const tableRows = ordered.concat(rows.filter(r => r.site === "All").map(r => Object.assign(r, {_cls: "total", _i: 1e6 + MODELS.findIndex(m => m.id === r.model)})));
  const model = id => MODELS.find(m => m.id === id) || {id, name: id, cls: "m1"};
  const cols = [
    {k: "site", label: "Site", left: true, sort: r => r._i, cell: r => esc(r.site)},
    {k: "model", label: "Model", left: true, sort: r => MODELS.findIndex(m => m.id === r.model), cell: r => modelCell(model(r.model))},
    ...mcols.map(([k, label]) => ({k, label, cell: r => {
      const ci = k === "cmAP" && num(r.cmAP_lo) ? ` <span class="ci-txt">${f2(r.cmAP_lo)}–${f2(r.cmAP_hi)}</span>` : "";
      return `<td class="${num(r[k]) && r[k] === best[r.site + "|" + k] ? "best" : ""}">${f3(r[k])}${ci}</td>`;
    }})),
    {k: "n_species", label: "Species", cell: r => String(r.n_species)},
    {k: "n_segs", label: "Segments", cell: r => Number(r.n_segs).toLocaleString("en-US")},
  ];
  const lede = scape
    ? `Soundscape mode: the full recordings, scored per 5-second segment and per recording unit. Dots show recording-level cmAP: model scores pooled per recording, compared with the species present.`
    : `Clips mode, BirdSet's protocol: each site's 5-second test clips, every species in the site's label set. Dots are segment cmAP; lines its 95% bootstrap interval over recordings.`;
  const allNote = allRow ? `<p class="meta">“All” is the plain mean of the per-site values over ${esc(allRow.sites.replaceAll(" ", ", "))}, so each site counts once whatever its size. Its species and segment counts are sums. Click a column heading to sort.</p>` : "";
  return `<p class="lede">${lede} The best value per site is bold.</p>${legend()}${chart}${table("overview", cols, tableRows)}${allNote}`;
}

// ---- grouped views ------------------------------------------------------------------
const GROUPING = {
  habitat: ["By species habitat", "Mean per-species AP, grouped by each species’ primary habitat in AVONET (Tobias et al. 2022). Habitat describes the bird, not the site. Habitats with fewer than 5 pairs in the current selection merge into Other."],
  biome: ["By site biome", "Mean per-species AP, grouped by the WWF biome of the recording site. With eight sites, each biome holds one to three sites, so a biome’s result is largely its sites’ results."],
  prevalence: ["By prevalence", "Mean per-species AP, grouped by how many labelled 5-second segments the species has at its site. Bins were fixed before looking at results."],
};

function grouped() {
  const [title, lede] = GROUPING[state.view];
  if (state.view === "biome" && state.site !== "All")
    return `<h3>${title}</h3><div class="empty">A single site has a single biome. Choose “All datasets” to compare biomes.</div>`;
  const rows = D.groups.filter(r => r.mode === state.mode && r.scope === state.site && r.grouping === state.view);
  if (!rows.length) return `<h3>${title}</h3><div class="empty">No results for this selection yet.</div>`;
  const order = [...new Set(rows.map(r => r.group))];
  const wide = order.map((g, i) => {
    const sub = rows.filter(r => r.group === g);
    const w = {group: g, _i: i, n_pairs: sub[0].n_pairs, sites: sub[0].sites};
    for (const r of sub) w[r.model] = r;
    return w;
  });
  const chart = dotChart(wide.map(w => ({key: w.group, label: w.group, sub: `${w.n_pairs} pairs`})),
    (g, m) => { const r = wide.find(w => w.group === g.key)[m.id]; return r && {v: r.mean_ap, lo: r.ap_lo, hi: r.ap_hi}; },
    "Mean per-species AP");
  const cols = [
    {k: "group", label: "Group", left: true, sort: r => r._i, cell: r => esc(r.group)},
    {k: "n_pairs", label: "Pairs", cell: r => String(r.n_pairs)},
    ...MODELS.map(m => ({k: m.id, label: esc(m.name), sort: r => r[m.id] && r[m.id].mean_ap, cell: r => {
      const x = r[m.id];
      if (!x) return "<td>–</td>";
      const best = Math.max(...MODELS.map(n => r[n.id] ? r[n.id].mean_ap : -1));
      return `<td class="${x.mean_ap === best ? "best" : ""}">${f3(x.mean_ap)} <span class="ci-txt">${f2(x.ap_lo)}–${f2(x.ap_hi)}</span></td>`;
    }})),
  ];
  if (state.site === "All") cols.push({k: "sites", label: "Sites", left: true, cell: r => esc(r.sites.replaceAll(" ", ", "))});
  const unit = state.mode === "scape" ? "segment AP within the soundscapes" : "AP on the test clips";
  return `<h3>${title}</h3><p class="lede">${lede} Each unit is one (site, species) pair, scored by ${unit}; intervals resample those pairs. Best model per group in bold.</p>${legend()}${chart}${table("g-" + state.view, cols, wide)}`;
}

// ---- species table -------------------------------------------------------------------
function metrics() {
  return state.mode === "scape"
    ? {ap: ["Segment AP", "AP per 5-second segment within the soundscapes"],
       auroc: ["Segment AUROC", "AUROC per 5-second segment"],
       rec_ap: ["Recording AP", "AP for presence per recording unit"],
       recall90: ["Recall @90% precision", "Share of recordings with the species that are found at the threshold giving 90% precision"]}
    : {ap: ["AP", "Average precision over the site's test clips"],
       auroc: ["AUROC", "Chance a clip with the species outscores one without"],
       recall90: ["Recall @90% precision", "Share of the species' clips found at the threshold giving 90% precision"]};
}

function speciesShell() {
  const ms = metrics();
  if (!ms[state.metric]) state.metric = "ap";
  const opt = (v, label, cur) => `<option value="${v}"${v === cur ? " selected" : ""}>${esc(label)}</option>`;
  return `<h3>By species</h3>
  <p class="lede">One row per site and species, so a species found at three sites appears three times. Average precision for a species with only a few positives is close to chance, so species with fewer than ${RARE} labelled segments are hidden unless you show them.</p>
  <div class="subcontrols">
    <div class="ctl"><label class="ctl-label" for="sp-metric">Metric</label><select id="sp-metric">${Object.entries(ms).map(([k, [l]]) => opt(k, l, state.metric)).join("")}</select></div>
    <div class="ctl"><span class="ctl-label">Difference</span><span style="display:flex;gap:6px;align-items:center">
      <select id="sp-dA" aria-label="First model">${MODELS.map(m => opt(m.id, m.name, state.dA)).join("")}</select>−
      <select id="sp-dB" aria-label="Second model">${MODELS.map(m => opt(m.id, m.name, state.dB)).join("")}</select></span></div>
    <div class="ctl"><label class="ctl-label" for="sp-q">Search</label><input id="sp-q" type="search" placeholder="Species, family or habitat" value="${esc(state.q)}"></div>
    <label class="check"><input id="sp-rare" type="checkbox"${state.hideRare ? " checked" : ""}> Hide species with fewer than ${RARE} positives</label>
  </div>
  ${legend()}
  <div id="sp-body"></div>`;
}

function speciesBody() {
  const scape = state.mode === "scape";
  const k = state.metric;
  const all = D.species.filter(r => r.mode === state.mode && (state.site === "All" || r.site === state.site));
  const q = state.q.trim().toLowerCase();
  const matched = q ? all.filter(r => [r.species, r.family, r.habitat].some(v => v && v.toLowerCase().includes(q))) : all;
  const rows = state.hideRare ? matched.filter(r => r.n_pos >= RARE) : matched;
  const hidden = matched.length - rows.length;
  for (const r of rows) {
    const a = r[`${k}_${state.dA}`], b = r[`${k}_${state.dB}`];
    r._delta = num(a) && num(b) ? a - b : null;
    const vals = MODELS.map(m => r[`${k}_${m.id}`]).filter(num);
    r._best = vals.length ? Math.max(...vals) : null;
  }
  const nameA = MODELS.find(m => m.id === state.dA).name, nameB = MODELS.find(m => m.id === state.dB).name;
  const cols = [];
  if (state.site === "All") cols.push({k: "site", label: "Site", left: true, sort: r => r.site, cell: r => esc(r.site)});
  cols.push(
    {k: "species", label: "Species", left: true, cell: r => `<td class="l"><span class="sp">${esc(cap(r.species))}</span>${r.family ? ` <span class="fam">${esc(r.family)}</span>` : ""}</td>`},
    {k: "habitat", label: "Habitat", left: true, cell: r => esc(r.habitat || "Unknown")},
    {k: "n_pos", label: "Positives", title: "Labelled 5-second segments with the species", cell: r => `<td class="${r.n_pos < RARE ? "dim" : ""}">${r.n_pos}</td>`});
  if (scape) cols.push({k: "n_present", label: "Recordings", title: "Recording units where the species is present", cell: r => num(r.n_present) ? String(r.n_present) : "–"});
  for (const m of MODELS) cols.push({k: `${k}_${m.id}`, label: esc(m.name), cell: r => {
    const v = r[`${k}_${m.id}`];
    return `<td class="${num(v) && v === r._best ? "best" : ""}">${f3(v)}</td>`;
  }});
  cols.push({k: "_delta", label: "Δ", title: `${nameA} − ${nameB}`, cell: r => sgn(r._delta)});
  const count = `<p class="meta">${rows.length} of ${all.length} rows shown` +
    (q ? `, ${all.length - matched.length} not matching “${esc(state.q)}”` : "") +
    (hidden ? `, ${hidden} hidden with fewer than ${RARE} positives` : "") +
    `. Δ is ${esc(nameA)} − ${esc(nameB)} on ${esc(metrics()[k][0])}. Best model per row in bold.</p>`;
  if (!all.length) return `<div class="empty">No ${scape ? "soundscape" : "clips"} results for ${esc(state.site)} yet.</div>`;
  return count + (rows.length ? table("species", cols, rows, {tall: true}) : `<div class="empty">No species match.</div>`);
}

// ---- wiring --------------------------------------------------------------------------
const view = document.getElementById("view");

function syncControls() {
  document.querySelectorAll(".seg[data-key]").forEach(seg => seg.querySelectorAll("button").forEach(b => {
    b.setAttribute("aria-pressed", String(state[seg.dataset.key] === b.dataset.v));
    if (b.dataset.v === "biome") { b.disabled = state.site !== "All"; b.title = b.disabled ? "Needs All datasets" : ""; }
  }));
  document.getElementById("site").value = state.site;
}

function render() {
  if (state.view === "biome" && state.site !== "All") state.view = "habitat";
  syncControls();
  view.innerHTML = state.view === "overview" ? overview() : state.view === "species" ? speciesShell() : grouped();
  if (state.view === "species") document.getElementById("sp-body").innerHTML = speciesBody();
  save();
}

function rerenderPart() {
  if (state.view === "species") document.getElementById("sp-body").innerHTML = speciesBody();
  else render();
  save();
}

document.querySelectorAll(".seg[data-key]").forEach(seg => seg.addEventListener("click", e => {
  const b = e.target.closest("button");
  if (!b || b.disabled) return;
  state[seg.dataset.key] = b.dataset.v;
  render();
}));
document.getElementById("site").addEventListener("change", e => { state.site = e.target.value; render(); });
view.addEventListener("click", e => {
  const b = e.target.closest("button[data-sort]");
  if (!b) return;
  const numeric = !["site", "species", "habitat", "group", "model", "sites"].includes(b.dataset.sort);
  onSort(b.dataset.table, b.dataset.sort, numeric);
  rerenderPart();
});
view.addEventListener("change", e => {
  const id = e.target.id;
  if (id === "sp-metric") state.metric = e.target.value;
  else if (id === "sp-dA") state.dA = e.target.value;
  else if (id === "sp-dB") state.dB = e.target.value;
  else if (id === "sp-rare") state.hideRare = e.target.checked;
  else return;
  rerenderPart();
});
view.addEventListener("input", e => {
  if (e.target.id !== "sp-q") return;
  state.q = e.target.value;
  rerenderPart();
});
render();
"""


def model_name(m: str) -> str:
    return MODELS.get(m, (m, "m1"))[0]


def model_cls(m: str) -> str:
    return MODELS.get(m, (m, "m1"))[1]


def fmt(v, d: int = 3) -> str:
    return "–" if pd.isna(v) else f"{v:.{d}f}"


def static_table(head: list[str], body: list[list[str]], left_cols: int = 1) -> str:
    th = "".join(f'<th class="{"l" if i < left_cols else ""}">{h}</th>' for i, h in enumerate(head))
    trs = "".join("<tr>" + "".join(f'<td class="{"l" if i < left_cols else ""}">{c}</td>' for i, c in enumerate(r))
                  + "</tr>" for r in body)
    return f'<div class="table-wrap"><table><thead><tr>{th}</tr></thead><tbody>{trs}</tbody></table></div>'


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
    sites = ", ".join(sorted(p["site"].unique()))
    return f"""<section id="published">
  <h2>Check against published figures</h2>
  <p class="lede">Perch V2 in clips mode against the three Perch 2.0 variants in van Merriënboer et al. 2025
  (arXiv:2508.04665, Appendix C, Table 4). A value counts as within if it is no more than 0.02 outside the published
  range. {n_ok} of {len(p)} values on {escape(sites)} are within.</p>
  <details><summary>Show the table</summary>
  {static_table(["Site", "Metric", "Ours", "Published", "Distance", "Result"], body, left_cols=2)}
  </details>
  <p class="meta">POW is not in the Perch 2.0 table: BirdSet uses it as a validation set.
  BirdNET has no published per-site figures; BirdSet left it out because its training data may overlap the test sets.</p>
</section>"""


def takeaways(h: pd.DataFrame, p: pd.DataFrame, g: pd.DataFrame, models: list[str]) -> str:
    items = []
    clips = h[(h["mode"] == "clips")]
    allrow = clips[clips.site == "All"].set_index("model")
    per = clips[clips.site != "All"]
    if not allrow.empty:
        n = len(allrow["sites"].iloc[0].split())
        rank = allrow["cmAP"].sort_values(ascending=False)
        items.append(f"<li>Averaged over {n} sites, clips cmAP is "
                     + ", ".join(f"{escape(model_name(m))} {v:.3f}" for m, v in rank.items()) + ".</li>")
    wins: dict[str, list[str]] = {}
    overlap = []
    for s, sub in per.groupby("site"):
        sub = sub.sort_values("cmAP", ascending=False)
        if len(sub) < 2:
            continue
        a, b = sub.iloc[0], sub.iloc[1]
        wins.setdefault(a.model, []).append(s)
        if a.cmAP_lo <= b.cmAP_hi:
            overlap.append(s)
    if wins:
        parts = [f"{escape(model_name(m))} on {', '.join(sorted(ss))}" for m, ss in
                 sorted(wins.items(), key=lambda kv: -len(kv[1]))]
        items.append("<li>Highest cmAP per site: " + "; ".join(parts) + "."
                     + (f" The top two models' intervals overlap on {', '.join(sorted(overlap))}." if overlap else "")
                     + "</li>")
    scape = h[(h["mode"] == "scape") & (h.site == "All")].set_index("model")
    if not scape.empty:
        rank = scape["rec_recall90"].sort_values(ascending=False)
        items.append("<li>In the soundscapes, recall at 90% precision per recording unit (mean over sites) is "
                     + ", ".join(f"{escape(model_name(m))} {v:.2f}" for m, v in rank.items())
                     + ": the gap between models is larger than in cmAP.</li>")
    if not p.empty:
        n_ok = int((~p["flag"].astype(str).eq("True")).sum())
        items.append(f"<li>Perch V2 matches the Perch 2.0 paper: {n_ok} of {len(p)} published values on "
                     f"{len(p['site'].unique())} sites are reproduced within 0.02.</li>")
    ab = g[(g["mode"] == "clips") & (g.scope == "All") & (g.grouping == "prevalence")]
    if not ab.empty:
        first, last = ab.group.iloc[0], ab.group.iloc[-1]
        lo = ab[ab.group == first].set_index("model").mean_ap
        hi = ab[ab.group == last].set_index("model").mean_ap
        items.append(f"<li>Every model does better on species with many labelled segments ({escape(last)}: "
                     f"{hi.min():.2f}–{hi.max():.2f} mean AP) than on rare ones ({escape(first)}: "
                     f"{lo.min():.2f}–{lo.max():.2f}).</li>")
    return f'<ul class="takeaways">{"".join(items)}</ul>'


def records(df: pd.DataFrame) -> list[dict]:
    return json.loads(df.to_json(orient="records"))  # NaN -> null


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", type=Path, default=Path("results/report"))
    ap.add_argument("--sites", type=Path, default=Path("configs/reference/sites.csv"))
    ap.add_argument("--out", type=Path)
    ap.add_argument("--title", default="BirdSet Model Comparison")
    args = ap.parse_args()
    out = args.out or args.report / "index.html"

    h = pd.read_csv(args.report / "headline.csv")
    p = pd.read_csv(args.report / "published.csv")
    sp = pd.read_csv(args.report / "species.csv")
    g = pd.read_csv(args.report / "groups.csv")
    sites_df = pd.read_csv(args.sites)
    present = set(h.site) - {"All"}
    sites_df = sites_df[sites_df.site.isin(present)]
    models = [m for m in MODELS if m in set(h.model)] + sorted(set(h.model) - set(MODELS))
    sites = list(sites_df.site) + sorted(present - set(sites_df.site))
    today = dt.date.today().isoformat()

    data = {
        "models": [{"id": m, "name": model_name(m), "cls": model_cls(m)} for m in models],
        "sites": records(sites_df),
        "headline": records(h),
        "groups": records(g),
        "species": records(sp.drop(columns=["migration", "biome"], errors="ignore")),
    }
    blob = json.dumps(data, separators=(",", ":"), ensure_ascii=False).replace("</", "<\\/")
    site_opts = "".join(f'<option value="{escape(s)}">{escape(s)} · {escape(n)}</option>'
                        for s, n in zip(sites_df.site, sites_df.name))
    scape_sites = sorted(set(h[h["mode"] == "scape"].site) - {"All"})
    clip_sites = sorted(set(h[h["mode"] == "clips"].site) - {"All"})

    page = f"""<meta charset="utf-8">
<title>{escape(args.title)}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans+Condensed:wght@600&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap">
<style>{CSS}</style>
<main>
<header>
  <div class="eyebrow">birdeval · BirdSet {" · ".join(sites)}</div>
  <h1>{escape(args.title)}</h1>
  <p class="lede">{", ".join(escape(model_name(m)) for m in models)} on {len(sites)} BirdSet test sites.
  Clips mode follows BirdSet's protocol: every species in the site's label set, 5-second test clips
  ({len(clip_sites)} sites). Soundscape mode scores the full recordings ({len(scape_sites)} sites).</p>
  <p class="meta">Generated {today} by scripts/render_report.py</p>
</header>
<section id="summary">
  <h2>Summary</h2>
  {takeaways(h, p, g, models)}
</section>
<section id="explore">
  <h2>Explore the results</h2>
  <div class="controls" role="group" aria-label="Report filters">
    <div class="ctl"><span class="ctl-label" id="l-mode">Mode</span>
      <div class="seg" data-key="mode" aria-labelledby="l-mode">
        <button data-v="clips">Clips</button><button data-v="scape">Soundscapes</button></div></div>
    <div class="ctl"><label class="ctl-label" for="site">Dataset</label>
      <select id="site"><option value="All">All datasets</option>{site_opts}</select></div>
    <div class="ctl"><span class="ctl-label" id="l-view">View</span>
      <div class="seg" data-key="view" aria-labelledby="l-view">
        <button data-v="overview">Overview</button><button data-v="habitat">Habitat</button><button data-v="biome">Site biome</button><button data-v="prevalence">Prevalence</button><button data-v="species">Species</button></div></div>
  </div>
  <div id="view" aria-live="polite"></div>
</section>
{published_section(p)}
<section id="notes">
  <h2>Notes</h2>
  <ul class="notes">
    <li>BirdNET may have been trained on recordings from these test sets, so its scores may be optimistic.
    BirdSet excluded it for this reason.</li>
    <li>Species names follow the 2025 eBird taxonomy. Older model names are mapped through
    configs/taxonomy/synonyms.yaml. Every species at every site is scored for every model, including the 4 species
    BirdNET 2.4 cannot predict, which count against it.</li>
    <li>Per-species AP is noisy for species with only a few labelled segments; group means pool many species but
    their intervals still widen as groups get small.</li>
    <li>Site biomes are WWF biomes assigned per site (configs/reference/sites.csv); the sites are small areas, so
    a label describes the site's setting, not every recorder.</li>
    <li>POW is BirdSet's validation set, not a test set. It is included here like the others.</li>
  </ul>
  <details><summary>Metric definitions</summary>
  <dl>
    <dt>cmAP</dt><dd>Average precision per species, averaged over species. Treats rare and common species equally.</dd>
    <dt>AUROC</dt><dd>Chance a positive segment scores higher than a negative one, averaged over species.</dd>
    <dt>Top-1</dt><dd>Share of segments with at least one label where the top-scoring species is one of them.</dd>
    <dt>Recording cmAP</dt><dd>cmAP for species presence per recording unit, from soundscape mode.</dd>
    <dt>Recall @90%P</dt><dd>At the single score threshold where 90% of a model's claims are correct, the share of
    true presences it finds. Per clip in clips mode, per recording unit in soundscape mode.</dd>
    <dt>Positives</dt><dd>Labelled 5-second segments with the species at that site.</dd>
  </dl>
  </details>
</section>
<footer>Habitat data: AVONET, Tobias et al. 2022, Ecology Letters 25:581 (CC BY 4.0).
Published figures: van Merriënboer et al. 2025 (Perch 2.0); Rauch et al. 2024 (BirdSet, arXiv:2403.10380).</footer>
</main>
<script type="application/json" id="report-data">{blob}</script>
<script>{JS}</script>
"""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page)
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()

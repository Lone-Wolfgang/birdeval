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
.name { white-space: normal; min-width: 14em; }
button.exp { font: inherit; color: var(--fg); background: none; border: 0; padding: 0; cursor: pointer; display: inline-flex; gap: 6px; align-items: baseline; text-align: left; }
button.exp:hover .cn { text-decoration: underline; text-underline-offset: 3px; }
.tw { color: var(--muted); font-size: 11px; width: 1ch; }
.cn { font-weight: 600; }
.sp { margin-left: 1.6ch; }
.links { display: inline-flex; gap: 4px; margin-left: 8px; vertical-align: 1px; }
a.ext { font: 600 10.5px/1 var(--mono); color: var(--accent); text-decoration: none; border: 1px solid var(--rule); border-radius: 4px; padding: 3px 5px; }
a.ext:hover { border-color: var(--accent); }
a.ext:focus-visible { outline: 2px solid var(--accent); outline-offset: 1px; }
tr.detail > td { background: var(--bg); white-space: normal; text-align: left; padding: 16px 18px; }
tbody tr.detail:hover td { background: var(--bg); }
/* the detail row spans a table that may scroll sideways: keep the panel in the visible part */
.detail-grid { position: sticky; left: 18px; width: min(880px, calc(100vw - 92px)); display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(0, 384px); gap: 16px 28px; }
@media (max-width: 760px) { .detail-grid { grid-template-columns: minmax(0, 1fr); width: calc(100vw - 72px); } }
dl.facts { grid-template-columns: max-content minmax(0, 1fr); font-size: 13.5px; gap: 4px 14px; }
dl.facts dt { font: 600 11.5px/1.8 var(--mono); color: var(--muted); text-transform: uppercase; letter-spacing: .04em; }
.minis { display: flex; flex-direction: column; gap: 14px; min-width: 0; }
.minis .ctl-label { margin-bottom: 6px; }
svg.mini { display: block; width: 100%; max-width: 384px; height: auto; }
svg.mini text { font-family: var(--body); }
.bar { fill: var(--accent); }
.centre { stroke: var(--fg); stroke-width: 2; }
.gap { fill: url(#hatch); }
.hcol { fill: transparent; }
.gridline.zero { stroke: var(--muted); }
.hcol:hover { fill: var(--hover); opacity: .5; }
.hours-head { display: flex; flex-wrap: wrap; gap: 12px 24px; align-items: flex-end; justify-content: space-between; }
.hatch { stroke: var(--rule); stroke-width: 2; }
details.colpick { border: 0; padding: 0; position: relative; align-self: flex-end; }
details.colpick summary { font-size: 14px; color: var(--fg); border: 1px solid var(--rule); border-radius: 6px; padding: 7px 10px; background: var(--surface); }
details.colpick[open] summary { margin-bottom: 8px; }
.colgrid { display: grid; grid-template-columns: repeat(auto-fill, minmax(9em, 1fr)); gap: 6px 14px; padding: 10px 12px; background: var(--surface); border: 1px solid var(--rule); border-radius: 6px; }
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
const VIEWS = ["overview", "groups", "species", "hours"];
const RARE = 10;
const KEY = "birdeval-report-v2";

const state = Object.assign(
  {mode: "clips", site: "All", view: "overview", groupBy: "habitat", metric: "ap", dA: MODELS[0].id,
   dB: MODELS[MODELS.length - 1].id, hMetric: "AUROC", q: "", hideRare: true, sort: {}, cols: [], open: null},
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
  }).join("") + "</tr>" +
    (opts.detail && opts.isOpen(r) ? `<tr class="detail"><td colspan="${cols.length}">${opts.detail(r)}</td></tr>` : "")).join("");
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
// key -> [menu label, heading, lede]
const GROUPING = {
  habitat: ["Habitat", "By species habitat", "Grouped by each species’ primary habitat in AVONET (Tobias et al. 2022). Habitat describes the bird, not the site."],
  biome: ["Site biome", "By site biome", "Grouped by the WWF biome of the recording site. With eight sites, each biome holds one to three sites, so a biome’s result is largely its sites’ results."],
  prevalence: ["Prevalence", "By prevalence", "Grouped by how many labelled 5-second segments the species has at its site."],
  mass: ["Body mass", "By body mass", "Grouped by AVONET body mass."],
  lifestyle: ["Lifestyle", "By lifestyle", "Grouped by AVONET primary lifestyle: aerial, insessorial (perching), terrestrial, aquatic or generalist."],
  stratum: ["Foraging stratum", "By foraging stratum", "Grouped by the stratum where the species does most of its foraging, from EltonTraits 1.0 (Wilman et al. 2014)."],
  diet: ["Diet", "By diet", "Grouped by EltonTraits’ five diet categories."],
  call_freq: ["Call frequency", "By call frequency", "Grouped by the median centre frequency of the species’ annotated calls at its site, from BirdSet’s call boxes. NBP has no call boxes."],
  xc: ["Xeno-canto recordings", "By xeno-canto recordings", "Grouped by how many xeno-canto recordings the species has: a rough proxy for how much training audio the models had for it."],
};

function grouped() {
  const avail = Object.keys(GROUPING).filter(g => D.groups.some(r => r.mode === state.mode && r.grouping === g));
  if (!avail.includes(state.groupBy)) state.groupBy = avail[0] || "habitat";
  const [, title, lede] = GROUPING[state.groupBy];
  const picker = `<div class="subcontrols"><div class="ctl"><label class="ctl-label" for="g-by">Group by</label>
    <select id="g-by">${avail.map(g => `<option value="${g}"${g === state.groupBy ? " selected" : ""}${g === "biome" && state.site !== "All" ? " disabled" : ""}>${esc(GROUPING[g][0])}${g === "biome" && state.site !== "All" ? " (all datasets only)" : ""}</option>`).join("")}</select></div></div>`;
  if (state.groupBy === "biome" && state.site !== "All")
    return `${picker}<h3>${title}</h3><div class="empty">A single site has a single biome. Choose “All datasets” to compare biomes.</div>`;
  const rows = D.groups.filter(r => r.mode === state.mode && r.scope === state.site && r.grouping === state.groupBy);
  if (!rows.length) return `${picker}<h3>${title}</h3><div class="empty">No results for this selection yet.</div>`;
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
  const binned = ["prevalence", "mass", "call_freq", "xc"].includes(state.groupBy);
  return `${picker}<h3>${title}</h3><p class="lede">Mean per-species AP. ${lede} ` +
    (binned ? "Bins were fixed before looking at results. " : "Categories with fewer than 5 pairs in the current selection merge into Other. ") +
    `Each unit is one (site, species) pair, scored by ${unit}; intervals resample those pairs. Best model per group in bold.</p>${legend()}${chart}${table("g-" + state.groupBy, cols, wide)}`;
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

// optional columns: key -> [label, title, cell(row), sort(row)]
const pct = v => num(v) ? Math.round(v * 100) + "%" : "–";
const EXTRA = {
  mass_g: ["Mass", "AVONET body mass (g)", r => num(r.mass_g) ? (r.mass_g < 10 ? r.mass_g.toFixed(1) : Math.round(r.mass_g).toLocaleString("en-US")) + " g" : "–"],
  lifestyle: ["Lifestyle", "AVONET primary lifestyle", r => esc(r.lifestyle || "–")],
  stratum: ["Foraging", "EltonTraits main foraging stratum", r => esc(r.stratum || "–")],
  diet: ["Diet", "EltonTraits diet category", r => esc(r.diet || "–")],
  nocturnal: ["Nocturnal", "EltonTraits nocturnal flag", r => r.nocturnal === true ? "Yes" : r.nocturnal === false ? "No" : "–", r => r.nocturnal === true ? 1 : r.nocturnal === false ? 0 : null],
  freq_centre_khz: ["Call band", "Median low–high frequency of the species' annotated calls at this site", r => num(r.freq_low_khz) ? `${r.freq_low_khz.toFixed(1)}–${r.freq_high_khz.toFixed(1)} kHz` : "–"],
  night_share: ["Night calls", "Share of annotated calls between 20:00 and 05:00 local time (only where the site recorded at night)", r => pct(r.night_share)],
  xc_total: ["XC recs", "Xeno-canto recordings of the species", r => num(r.xc_total) ? r.xc_total.toLocaleString("en-US") : "–"],
  xc_site_countries: ["XC local", "Xeno-canto recordings from the site's country", r => num(r.xc_site_countries) ? r.xc_site_countries.toLocaleString("en-US") : "–"],
};
const EXTRA_AVAIL = Object.keys(EXTRA).filter(k => D.species.some(r => r[k] !== null && r[k] !== undefined));
const rowKey = r => `${r.site}|${r.species}`;
const xcUrl = r => `https://xeno-canto.org/species/${encodeURIComponent(cap(r.species.split(" ").slice(0, 2).join("-")))}`;
const links = r => `<span class="links"><a class="ext" href="${xcUrl(r)}" target="_blank" rel="noopener" title="Recordings on xeno-canto">XC</a>` +
  (r.ebird_code ? `<a class="ext" href="https://ebird.org/species/${encodeURIComponent(r.ebird_code)}" target="_blank" rel="noopener" title="Species page on eBird">eBird</a>` : "") + "</span>";

function activitySvg(r) {
  const eff = D.effort[r.site];
  if (!r.hours || !eff) return `<p class="meta">No call times for this site.</p>`;
  const n = r.hours.split(";").map(Number);
  const rate = n.map((c, h) => eff[h] > 0 ? c / (eff[h] / 60) : null);
  const max = Math.max(...rate.filter(num), 1e-9);
  const bw = 16, H = 56, top = 6;
  const bars = rate.map((v, h) => {
    const x = h * bw;
    const tip = eff[h] > 0 ? `${String(h).padStart(2, "0")}:00 · ${n[h]} calls in ${Math.round(eff[h])} recorded min (${v.toFixed(1)} per hour)` : `${String(h).padStart(2, "0")}:00 · not recorded`;
    if (v === null) return `<rect class="gap" x="${x + 1}" y="${top}" width="${bw - 2}" height="${H}"><title>${tip}</title></rect>`;
    const bh = Math.max(v > 0 ? 2 : 0, v / max * H);
    return `<g><title>${tip}</title><rect x="${x}" y="${top}" width="${bw}" height="${H}" fill="transparent"/><rect class="bar" x="${x + 2}" y="${top + H - bh}" width="${bw - 4}" height="${bh}" rx="2"/></g>`;
  }).join("");
  const ticks = [0, 6, 12, 18, 24].map(h => `<text class="tick" x="${h * bw}" y="${top + H + 14}" text-anchor="${h === 0 ? "start" : h === 24 ? "end" : "middle"}">${String(h).padStart(2, "0")}</text>`).join("");
  return `<svg class="mini" viewBox="0 0 ${24 * bw} ${top + H + 18}" role="img" aria-label="Calls per recorded hour by local time">
    <defs><pattern id="hatch" width="5" height="5" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><line x1="0" y1="0" x2="0" y2="5" class="hatch"/></pattern></defs>
    <line class="gridline" x1="0" x2="${24 * bw}" y1="${top + H}" y2="${top + H}"/>${bars}${ticks}</svg>`;
}

function freqSvg(r) {
  if (!num(r.freq_low_khz)) return `<p class="meta">No call boxes for this site.</p>`;
  const top = Math.max(16, Math.ceil(r.freq_high_khz)), W = 384, x = v => v / top * W;
  const ticks = [0, 4, 8, 12, 16].filter(t => t <= top).map(t => `<text class="tick" x="${x(t)}" y="34" text-anchor="${t === 0 ? "start" : t === top ? "end" : "middle"}">${t}</text>`).join("");
  return `<svg class="mini" viewBox="0 0 ${W} 38" role="img" aria-label="Call frequency band">
    <line class="gridline" x1="0" x2="${W}" y1="12" y2="12"/>
    <rect class="bar" x="${x(r.freq_low_khz)}" y="4" width="${Math.max(2, x(r.freq_high_khz) - x(r.freq_low_khz))}" height="16" rx="3" opacity=".55"><title>${r.freq_low_khz.toFixed(1)}–${r.freq_high_khz.toFixed(1)} kHz (medians)</title></rect>
    <line class="centre" x1="${x(r.freq_centre_khz)}" x2="${x(r.freq_centre_khz)}" y1="2" y2="22"/>${ticks}</svg>`;
}

function detail(r) {
  const fact = (k, v) => v && v !== "–" ? `<dt>${k}</dt><dd>${v}</dd>` : "";
  const mass = EXTRA.mass_g[2](r);
  const site = siteName(r.site);
  return `<div class="detail-grid">
    <dl class="facts">
      ${fact("Family", esc(r.family))}
      ${fact("Habitat", esc([r.habitat, r.habitat_density && r.habitat_density !== "Unknown" ? r.habitat_density.toLowerCase() : ""].filter(Boolean).join(", ")))}
      ${fact("Body mass", mass + (num(r.wing_mm) ? `, wing ${Math.round(r.wing_mm)} mm` : ""))}
      ${fact("Lifestyle", esc(r.lifestyle))}
      ${fact("Diet", esc([r.diet, r.trophic_niche && r.trophic_niche !== r.diet ? `AVONET niche: ${r.trophic_niche.toLowerCase()}` : ""].filter(Boolean).join("; ")))}
      ${fact("Foraging", esc(r.stratum))}
      ${fact("Nocturnal", r.nocturnal === true ? "Yes (EltonTraits)" : r.nocturnal === false ? "No" : "")}
      ${fact("Migration", esc(r.migration))}
      ${fact("Xeno-canto", num(r.xc_total) ? `${r.xc_total.toLocaleString("en-US")} recordings` + (num(r.xc_site_countries) ? `, ${r.xc_site_countries.toLocaleString("en-US")} from the site's country` : "") : "")}
      ${fact(`At ${esc(r.site)}`, num(r.n_events) ? `${r.n_events.toLocaleString("en-US")} annotated calls, ${r.n_pos} labelled segments` : `${r.n_pos} labelled segments`)}
      ${fact("Links", links(r) + ` <a class="ext" href="https://search.macaulaylibrary.org/catalog?taxonCode=${encodeURIComponent(r.ebird_code || "")}&mediaType=audio" target="_blank" rel="noopener" title="Audio in the Macaulay Library">Macaulay</a>`)}
    </dl>
    <div class="minis">
      <div><div class="ctl-label">Calls per recorded hour at ${esc(r.site)} · local time</div>${activitySvg(r)}
        <p class="meta">Hatched hours were never recorded at ${esc(site)}.</p></div>
      <div><div class="ctl-label">Call frequency band at ${esc(r.site)} · kHz</div>${freqSvg(r)}</div>
    </div>
  </div>`;
}

function speciesShell() {
  const ms = metrics();
  if (!ms[state.metric]) state.metric = "ap";
  const opt = (v, label, cur) => `<option value="${v}"${v === cur ? " selected" : ""}>${esc(label)}</option>`;
  return `<h3>By species</h3>
  <p class="lede">One row per site and species, so a species found at three sites appears three times. Click a name for its traits, calling times and call frequency at that site. Average precision for a species with only a few positives is close to chance, so species with fewer than ${RARE} labelled segments are hidden unless you show them.</p>
  <div class="subcontrols">
    <div class="ctl"><label class="ctl-label" for="sp-metric">Metric</label><select id="sp-metric">${Object.entries(ms).map(([k, [l]]) => opt(k, l, state.metric)).join("")}</select></div>
    <div class="ctl"><span class="ctl-label">Difference</span><span style="display:flex;gap:6px;align-items:center">
      <select id="sp-dA" aria-label="First model">${MODELS.map(m => opt(m.id, m.name, state.dA)).join("")}</select>−
      <select id="sp-dB" aria-label="Second model">${MODELS.map(m => opt(m.id, m.name, state.dB)).join("")}</select></span></div>
    <div class="ctl"><label class="ctl-label" for="sp-q">Search</label><input id="sp-q" type="search" placeholder="Name, family or habitat" value="${esc(state.q)}"></div>
    <label class="check"><input id="sp-rare" type="checkbox"${state.hideRare ? " checked" : ""}> Hide species with fewer than ${RARE} positives</label>
    <details class="colpick"><summary>Columns${state.cols.length ? ` (+${state.cols.length})` : ""}</summary>
      <div class="colgrid">${EXTRA_AVAIL.map(k => `<label class="check"><input type="checkbox" data-col="${k}"${state.cols.includes(k) ? " checked" : ""}> <span title="${esc(EXTRA[k][1])}">${esc(EXTRA[k][0])}</span></label>`).join("")}</div>
    </details>
  </div>
  ${legend()}
  <div id="sp-body"></div>`;
}

function speciesBody() {
  const scape = state.mode === "scape";
  const k = state.metric;
  const all = D.species.filter(r => r.mode === state.mode && (state.site === "All" || r.site === state.site));
  const q = state.q.trim().toLowerCase();
  const matched = q ? all.filter(r => [r.species, r.common_name, r.family, r.habitat].some(v => v && v.toLowerCase().includes(q))) : all;
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
    {k: "species", label: "Species", left: true, sort: r => r.common_name || r.species, cell: r => {
      const open = state.open === rowKey(r);
      return `<td class="l name"><button class="exp" data-row="${esc(rowKey(r))}" aria-expanded="${open}"><span class="tw">${open ? "▾" : "▸"}</span>` +
        `<span class="cn">${esc(r.common_name || cap(r.species))}</span></button>${links(r)}<br><span class="sp">${esc(cap(r.species))}</span>${r.family ? ` <span class="fam">${esc(r.family)}</span>` : ""}</td>`;
    }},
    {k: "habitat", label: "Habitat", left: true, cell: r => esc(r.habitat || "Unknown")},
    ...state.cols.filter(k => EXTRA[k]).map(k => ({k, label: EXTRA[k][0], title: EXTRA[k][1], left: ["lifestyle", "stratum", "diet"].includes(k), cell: EXTRA[k][2], sort: EXTRA[k][3]})),
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
  return count + (rows.length ? table("species", cols, rows, {tall: true, detail, isOpen: r => state.open === rowKey(r)}) : `<div class="empty">No species match.</div>`);
}


// ---- by hour of day ---------------------------------------------------------------------
// key -> [menu label, axis label, unit is 0-1, higher is better, description]
const HMETRIC = {
  AUROC: ["AUROC", "AUROC", true, true, "Chance a segment with the species scores higher than one without, averaged over the species heard in that hour. It does not depend on how common calls are, so it compares fairly across busy and quiet hours."],
  recall: ["Recall at 90% precision", "Recall", true, true, "Share of (segment, species) presences the model finds, with each species’ threshold fixed so that 90% of its claims are right over the site’s whole test set."],
  false_alarms_h: ["False alarms per hour", "False alarms per hour of audio", false, false, "Wrong (segment, species) claims per hour of audio at the same fixed thresholds: how often the model cries wolf at that time of day."],
  precision: ["Precision at fixed thresholds", "Precision", true, true, "Share of claims that are right in that hour, at thresholds set for 90% precision over the whole site."],
  cmAP: ["cmAP", "cmAP", true, true, "Average precision per species, averaged. AP falls when calls are rare, so quiet hours score lower even for an unchanged model: compare with care."],
  top1: ["Top-1", "Top-1", true, true, "Share of segments with a bird where the top-scoring species is one that is there."],
};
const hh = h => String(h).padStart(2, "0");
const HW = 760, HL = 56, HR = 16, hx = h => HL + (h + .5) * (HW - HL - HR) / 24, hbw = (HW - HL - HR) / 24;

function hourAxis(y) {
  return [0, 3, 6, 9, 12, 15, 18, 21].map(h => `<text class="tick" x="${hx(h)}" y="${y}" text-anchor="middle">${hh(h)}</text>`).join("") +
    `<text class="tick" x="${HW - HR}" y="${y + 14}" text-anchor="end">local hour</text>`;
}

function activityChart(rows) {
  const H = 120, top = 18, base = top + 80;
  const by = Object.fromEntries(rows.map(r => [r.hour, r]));
  const y = v => base - v * 80;
  const out = [`<svg viewBox="0 0 ${HW} ${H + 16}" role="img" aria-label="Share of 5-second segments with a bird, by local hour">`,
    `<defs><pattern id="hatch-h" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><line x1="0" y1="0" x2="0" y2="6" class="hatch"/></pattern></defs>`,
    `<text class="tick" x="${HL}" y="10">Share of 5-second segments with a bird</text>`];
  for (const t of [0, .5, 1]) out.push(`<line class="gridline" x1="${HL}" x2="${HW - HR}" y1="${y(t)}" y2="${y(t)}"/>`,
    `<text class="tick" x="${HL - 8}" y="${y(t) + 4}" text-anchor="end">${Math.round(t * 100)}%</text>`);
  for (let h = 0; h < 24; h++) {
    const r = by[h], x = hx(h) - hbw / 2;
    if (!r) { out.push(`<rect fill="url(#hatch-h)" x="${x + 1}" y="${y(1)}" width="${hbw - 2}" height="80"><title>${hh(h)}:00 · not recorded (or under 30 min)</title></rect>`); continue; }
    const tip = `${hh(h)}:00 · ${Math.round(r.bird_share * 100)}% of segments with a bird, ${r.species_per_seg.toFixed(2)} species per segment, ${r.n_species} ${state.site === "All" ? "site × species pairs" : "species"} heard · ${Math.round(r.minutes)} min in ${r.n_files} recordings`;
    out.push(`<g><title>${esc(tip)}</title><rect x="${x}" y="${y(1)}" width="${hbw}" height="80" fill="transparent"/>` +
      `<rect class="bar" x="${x + 3}" y="${y(r.bird_share)}" width="${hbw - 6}" height="${Math.max(1, 80 * r.bird_share)}" rx="2"/></g>`);
  }
  out.push(hourAxis(base + 16), "</svg>");
  return `<div class="chart">${out.join("")}</div>`;
}

function metricChart(rows, k) {
  const [, axis0, unit, higher] = HMETRIC[k];
  const delta = rows.some(r => r.delta);
  const axis = delta ? `${axis0}: change from each site’s all-day value` : axis0;
  const vals = rows.flatMap(r => [r[k], r[k + "_lo"], r[k + "_hi"]]).filter(num);
  if (!vals.length) return `<div class="empty">No values for this metric.</div>`;
  let lo, hi;
  if (delta) {  // symmetric around zero
    const step0 = unit ? .05 : niceStep(Math.max(...vals.map(Math.abs)));
    hi = Math.max(step0, Math.ceil(Math.max(...vals.map(Math.abs)) / step0) * step0); lo = -hi;
  } else {
    lo = unit ? Math.max(0, Math.floor(Math.min(...vals) * 10) / 10) : 0;
    hi = unit ? Math.min(1, Math.ceil(Math.max(...vals) * 10) / 10) : Math.max(...vals) * 1.05;
    if (hi - lo < .2 && unit) lo = Math.max(0, hi - .2);
  }
  const H = 250, top = 22, base = top + 200;
  const y = v => base - (v - lo) / (hi - lo || 1) * 200;
  const step = delta ? niceStepFine(hi, unit) : unit ? (hi - lo > .5 ? .2 : .1) : niceStep(hi);
  const tfmt = t => delta ? (Math.abs(t) < 1e-9 ? "0" : (t > 0 ? "+" : "−") + (unit ? Math.abs(t).toFixed(2) : Math.round(Math.abs(t)))) : unit ? t.toFixed(1) : String(Math.round(t));
  const out = [`<svg viewBox="0 0 ${HW} ${H + 16}" role="img" aria-label="${esc(axis)} by local hour">`,
    `<text class="tick" x="${HL}" y="10">${esc(axis)}${higher ? " (higher is better)" : " (lower is better)"}${!delta && unit && lo > 0 ? " · axis starts at " + lo.toFixed(1) : ""}</text>`];
  for (let i = 0, t = lo; t <= hi + 1e-9; t = lo + ++i * step) out.push(`<line class="gridline${delta && Math.abs(t) < 1e-9 ? " zero" : ""}" x1="${HL}" x2="${HW - HR}" y1="${y(t)}" y2="${y(t)}"/>`,
    `<text class="tick" x="${HL - 8}" y="${y(t) + 4}" text-anchor="end">${tfmt(t)}</text>`);
  const hours = [...new Set(rows.map(r => r.hour))].sort((a, b) => a - b);
  MODELS.forEach((m, i) => {
    const off = (i - (MODELS.length - 1) / 2) * 5;
    const mr = Object.fromEntries(rows.filter(r => r.model === m.id).map(r => [r.hour, r]));
    let path = "", prev = null;
    for (const h of hours) {
      const r = mr[h];
      if (!r || !num(r[k])) { prev = null; continue; }
      path += `${prev === h - 1 ? "L" : "M"}${(hx(h) + off).toFixed(1)} ${y(r[k]).toFixed(1)}`;
      prev = h;
    }
    const marks = hours.filter(h => mr[h] && num(mr[h][k])).map(h => {
      const r = mr[h], cx = hx(h) + off;
      return (num(r[k + "_lo"]) ? `<line class="ci" x1="${cx}" x2="${cx}" y1="${y(r[k + "_lo"])}" y2="${y(r[k + "_hi"])}" opacity=".55"/>` : "") +
        `<circle class="dot" cx="${cx}" cy="${y(r[k])}" r="4"/>`;
    }).join("");
    out.push(`<g class="${m.cls}"><path d="${path}" fill="none" stroke="var(--c)" stroke-width="2" stroke-linejoin="round"/>${marks}</g>`);
  });
  // hover columns: one tooltip per hour with every model
  const fmt = v => !num(v) ? "–" : delta ? (v >= 0 ? "+" : "−") + (unit ? Math.abs(v).toFixed(3) : Math.abs(v).toFixed(1)) : unit ? f3(v) : v.toFixed(1);
  for (const h of hours) {
    const tip = `${hh(h)}:00\n` + MODELS.map(m => {
      const r = rows.find(r => r.model === m.id && r.hour === h);
      return r ? `${m.name}: ${fmt(r[k])}` + (num(r[k + "_lo"]) ? ` (95% CI ${fmt(r[k + "_lo"])}${delta ? " to " : "–"}${fmt(r[k + "_hi"])})` : "") : "";
    }).filter(Boolean).join("\n");
    out.push(`<rect class="hcol" x="${hx(h) - hbw / 2}" y="${top}" width="${hbw}" height="200"><title>${esc(tip)}</title></rect>`);
  }
  out.push(hourAxis(base + 16), "</svg>");
  return `<div class="chart">${out.join("")}</div>`;
}

function niceStepFine(hi, unit) {
  if (unit) return hi > .2 ? .1 : hi > .1 ? .05 : .025;
  return niceStep(hi * 2) / 2;
}

function niceStep(max) {
  const raw = max / 4, p = 10 ** Math.floor(Math.log10(raw || 1));
  return [1, 2, 5, 10].map(m => m * p).find(s => s >= raw);
}

function hoursView() {
  const all = D.hourly || [];
  const sites = [...new Set(all.map(r => r.site))].filter(s => s !== "All");
  if (!all.length) return `<div class="empty">No hourly results yet: run scripts/hourly.py.</div>`;
  const rows = all.filter(r => r.site === state.site);
  if (!rows.length) return `<div class="empty">No hourly results for ${esc(state.site)}${state.site === "NBP" ? ": its BirdSet metadata has no clock times" : ""}. Hourly results exist for ${esc(sites.join(", "))}${D.hourly_missing ? `; ${esc(D.hourly_missing)}` : ""}.</div>`;
  const k = HMETRIC[state.hMetric] ? state.hMetric : "AUROC";
  const [, , unit, , desc] = HMETRIC[k];
  const picker = `<div class="subcontrols"><div class="ctl"><label class="ctl-label" for="h-metric">Metric</label>
    <select id="h-metric">${Object.entries(HMETRIC).map(([v, [l]]) => `<option value="${v}"${v === k ? " selected" : ""}>${l}</option>`).join("")}</select></div></div>`;
  const act = rows.filter(r => r.model === MODELS[0].id);
  const scope = state.site === "All"
    ? `All sites with clock times (${esc(sites.join(", "))}). Sites record at different hours, and the hardest site (PER) records only at dawn, so pooling raw values would mostly show which sites record when. Instead, the performance chart shows, for each hour, how much each site’s value in that hour differs from the same site’s value over its whole day, averaged over the sites recorded then: above zero means that time of day is easier than usual. Activity bars pool the sites.`
    : `${esc(state.site)} · ${esc(siteName(state.site))}. Hours with under 30 minutes of reviewed audio are left out.`;
  const delta = rows.some(r => r.delta);
  const fmt = v => !num(v) ? "–" : delta ? (v >= 0 ? "+" : "−") + (unit ? Math.abs(v).toFixed(3) : Math.abs(v).toFixed(1)) : unit ? f3(v) : v.toFixed(1);
  const best = {};
  for (const r of rows) if (num(r[k])) {
    const b = best[r.hour];
    if (b === undefined || (HMETRIC[k][3] ? r[k] > b : r[k] < b)) best[r.hour] = r[k];
  }
  const trows = act.map(a => {
    const o = {hour: a.hour, minutes: a.minutes, n_files: a.n_files, bird_share: a.bird_share, n_species: a.n_species, sites: a.sites};
    for (const m of MODELS) { const r = rows.find(r => r.model === m.id && r.hour === a.hour); o[m.id] = r ? r[k] : null; o[m.id + "_lo"] = r && r[k + "_lo"]; o[m.id + "_hi"] = r && r[k + "_hi"]; }
    return o;
  });
  const cols = [
    {k: "hour", label: "Hour", left: true, cell: r => `${hh(r.hour)}:00`},
    {k: "minutes", label: "Minutes", cell: r => Math.round(r.minutes).toLocaleString("en-US")},
    {k: "n_files", label: "Recordings", cell: r => String(r.n_files)},
    {k: "bird_share", label: "With a bird", cell: r => Math.round(r.bird_share * 100) + "%"},
    {k: "n_species", label: state.site === "All" ? "Pairs" : "Species", cell: r => String(r.n_species)},
    ...(delta ? [{k: "sites", label: "Sites", left: true, cell: r => esc(r.sites)}] : []),
    ...MODELS.map(m => ({k: m.id, label: modelCell(m), cell: r =>
      `<td class="${num(r[m.id]) && r[m.id] === best[r.hour] ? "best" : ""}">${fmt(r[m.id])}` +
      (num(r[m.id + "_lo"]) ? ` <span class="ci-txt">${fmt(r[m.id + "_lo"])}${delta ? " to " : "–"}${fmt(r[m.id + "_hi"])}</span>` : "") + "</td>"})),
  ];
  return `<div class="hours-head"><h3>By hour of day</h3>${picker}</div>
    <p class="lede">Soundscape mode, whichever mode is selected: the clips carry no clock time. ${scope}</p>
    ${activityChart(act)}
    <p class="lede"><strong>${esc(HMETRIC[k][0])}.</strong> ${esc(desc)} Lines join consecutive recorded hours; whiskers are 95% intervals from resampling recordings within the hour.</p>
    ${legend()}${metricChart(rows, k)}${table("hours", cols, trows)}`;
}

// ---- wiring --------------------------------------------------------------------------
const view = document.getElementById("view");

function syncControls() {
  document.querySelectorAll(".seg[data-key]").forEach(seg => seg.querySelectorAll("button").forEach(b => {
    b.setAttribute("aria-pressed", String(state[seg.dataset.key] === b.dataset.v));
  }));
  document.getElementById("site").value = state.site;
}

function render() {
  syncControls();
  view.innerHTML = state.view === "overview" ? overview() : state.view === "species" ? speciesShell()
    : state.view === "hours" ? hoursView() : grouped();
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
  const x = e.target.closest("button.exp");
  if (x) {
    state.open = state.open === x.dataset.row ? null : x.dataset.row;
    rerenderPart();
    const again = [...view.querySelectorAll("button.exp")].find(b => b.dataset.row === x.dataset.row);
    if (again) again.focus();
    return;
  }
  const b = e.target.closest("button[data-sort]");
  if (!b) return;
  const numeric = !["site", "species", "habitat", "group", "model", "sites", "lifestyle", "stratum", "diet"].includes(b.dataset.sort);
  onSort(b.dataset.table, b.dataset.sort, numeric);
  rerenderPart();
});
view.addEventListener("change", e => {
  const id = e.target.id;
  if (id === "sp-metric") state.metric = e.target.value;
  else if (id === "sp-dA") state.dA = e.target.value;
  else if (id === "sp-dB") state.dB = e.target.value;
  else if (id === "sp-rare") state.hideRare = e.target.checked;
  else if (id === "g-by") state.groupBy = e.target.value;
  else if (id === "h-metric") state.hMetric = e.target.value;
  else if (e.target.dataset.col) {
    const k = e.target.dataset.col;
    state.cols = e.target.checked ? EXTRA_AVAIL.filter(c => c === k || state.cols.includes(c)) : state.cols.filter(c => c !== k);
    const sum = view.querySelector(".colpick summary");
    if (sum) sum.textContent = "Columns" + (state.cols.length ? ` (+${state.cols.length})` : "");
  }
  else return;
  rerenderPart();
});
view.addEventListener("input", e => {
  if (e.target.id !== "sp-q") return;
  state.q = e.target.value;
  rerenderPart();
});
if (!VIEWS.includes(state.view)) { state.groupBy = state.view; state.view = "groups"; }  // v2 saved views
if (GROUPING[tok]) { state.view = "groups"; state.groupBy = tok; }
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


def takeaways(h: pd.DataFrame, p: pd.DataFrame, g: pd.DataFrame, models: list[str],
              hourly: pd.DataFrame | None = None) -> str:
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
    if hourly is not None and (hourly.site == "All").any():
        a = hourly[hourly.site == "All"]
        by = a.groupby("hour")["AUROC"].mean()
        def hour_txt(hr: int) -> str:
            r = a[a.hour == hr]
            sign = lambda v: ("+" if v >= 0 else "−") + f"{abs(v):.2f}"
            n_cls, n_rec = int(r.n_classes.max()), int(r.n_files.iloc[0])
            return (f"{hr:02d}:00 ({sign(r.AUROC.min())} to {sign(r.AUROC.max())} across models; "
                    f"{escape(r.sites.iloc[0].replace(' ', ', '))}, {n_rec} recordings, {n_cls} species)")
        items.append(f"<li>By hour of day, against each site's own all-day AUROC, the models do worst at "
                     f"{hour_txt(int(by.idxmin()))} and best at {hour_txt(int(by.idxmax()))}. Hours other than "
                     f"dawn rest on one to three sites; see the Hours view.</li>")
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
    effort_f = args.sites.with_name("site_effort.csv")
    effort = ({s: g.sort_values("hour")["minutes"].round(1).tolist() for s, g in pd.read_csv(effort_f).groupby("site")}
              if effort_f.exists() else {})
    hourly_f = args.report / "hourly.csv"
    hourly = pd.read_csv(hourly_f) if hourly_f.exists() else None
    hourly_missing = None
    if hourly is not None:
        lacking = sorted(set(h[h["mode"] == "scape"].site) - {"All"} - set(hourly.site) - {"NBP"})
        hourly_missing = (f"{', '.join(lacking)} not yet (some predictions not cached)" if lacking else None)
    xc_f = args.sites.with_name("xc_counts.csv")
    xc_date = pd.read_csv(xc_f)["query_date"].max() if xc_f.exists() else None
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
        "species": records(sp.drop(columns=["biome"], errors="ignore")),
        "effort": effort,
        "hourly": records(hourly) if hourly is not None else [],
        "hourly_missing": hourly_missing,
    }
    blob = json.dumps(data, separators=(",", ":"), ensure_ascii=False).replace("</", "<\\/")
    site_opts = "".join(f'<option value="{escape(s)}">{escape(s)} · {escape(n)}</option>'
                        for s, n in zip(sites_df.site, sites_df.name))
    scape_sites = sorted(set(h[h["mode"] == "scape"].site) - {"All"})
    clip_sites = sorted(set(h[h["mode"] == "clips"].site) - {"All"})

    xc_note = (f" Xeno-canto counts were queried on {xc_date}: they are today's counts, larger than what any model "
               "trained on, and xeno-canto's IOC names can split or lump differently from eBird's." if xc_date else "")
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
  {takeaways(h, p, g, models, hourly)}
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
        <button data-v="overview">Overview</button><button data-v="groups">Groups</button><button data-v="species">Species</button><button data-v="hours">Hours</button></div></div>
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
    <li>Calling times and call frequency bands come from BirdSet's call-level annotations (scripts/site_calls.py).
    BirdSet's “local_time” is UTC for PER and SSW (their file names end in Z); those are converted to local time.
    Most sites record only around dawn (PER 05–06 h, POW and HSN about 05–09 h); SSW records all day and UHH 05–23 h,
    so only those two show full daily patterns. Activity is calls per recorded hour, and unrecorded hours are shown as
    unknown. NBP has no times or frequency boxes.</li>
    <li>Body mass, lifestyle and habitat are AVONET; diet, foraging stratum and the nocturnal flag are EltonTraits 1.0,
    matched to eBird names through AVONET's crosswalk.{xc_note}</li>
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
<footer>Traits: AVONET, Tobias et al. 2022, Ecology Letters 25:581 (CC BY 4.0); EltonTraits 1.0, Wilman et al. 2014,
Ecology 95:2027 (CC0). Common names and links: eBird/Clements taxonomy 2025; xeno-canto.org; Macaulay Library.
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

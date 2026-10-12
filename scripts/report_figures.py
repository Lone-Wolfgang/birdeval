"""Inline SVG figures for the report page that need no data: the pipeline and a worked example.

Both use the page's CSS classes and colour tokens, so they follow the light and dark themes.
"""

from __future__ import annotations

from html import escape

ARROW = ('<defs><marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
         'orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" class="arrowhead"/></marker></defs>')


def _box(x, y, w, h, lines, cls="pbox"):
    t = "".join(f'<text class="ptext" x="{x + w / 2}" y="{y + h / 2 + (i - (len(lines) - 1) / 2) * 15 + 4}" '
                f'text-anchor="middle">{escape(s)}</text>' for i, s in enumerate(lines))
    return f'<g class="{cls}"><rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6"/>{t}</g>'


def _arrow(points):
    d = "M" + " L".join(f"{x} {y}" for x, y in points)
    return f'<path class="parrow" d="{d}" marker-end="url(#arr)"/>'


def pipeline_svg() -> str:
    """eval set -> segments; model -> raw logits -> cache -> scorer -> align -> segment metrics;
    scorer -> aggregate -> recording-level metrics."""
    W, bw, bh, gap = 900, 134, 50, 18
    col = [4 + i * (bw + gap) for i in range(6)]
    r1, r2, r3 = 14, 112, 214
    o = [f'<svg viewBox="0 0 {W} 300" role="img" aria-label="The birdeval pipeline: an eval set gives labelled '
         f'5-second segments; a model gives raw logits per window, which are cached; a scorer, alignment to the '
         f'segments and segment metrics follow; scores are also aggregated per recording for recording-level '
         f'metrics. Changing a metric never re-runs inference.">', ARROW]
    o.append(_box(col[0], r1, bw, bh, ["BirdSet eval set", "(one site)"]))
    o.append(_box(col[4], r1, bw, bh, ["5-second segments", "and labels"]))
    o.append(_arrow([(col[0] + bw, r1 + bh / 2), (col[4] - 4, r1 + bh / 2)]))
    o.append(_box(col[0], r2, bw, bh, ["Model", "(BirdNET, Perch)"]))
    o.append(_box(col[1], r2, bw, bh, ["Raw logits", "per window"]))
    o.append(_box(col[2], r2, bw, bh, ["Prediction", "cache"], cls="pbox key"))
    o.append(_box(col[3], r2, bw, bh, ["Scorer", "(logit or sigmoid)"]))
    o.append(_box(col[4], r2, bw, bh, ["Align to segments", "(half-window rule)"]))
    o.append(_box(col[5], r2, bw, bh, ["Segment metrics", "cmAP · AUROC · top-1"]))
    for i in range(5):
        o.append(_arrow([(col[i] + bw, r2 + bh / 2), (col[i + 1] - 4, r2 + bh / 2)]))
    o.append(_arrow([(col[4] + bw / 2, r1 + bh), (col[4] + bw / 2, r2 - 4)]))
    o.append(_box(col[4], r3, bw, bh, ["Aggregate", "per recording"]))
    o.append(_box(col[5], r3, bw, bh, ["Recording-level", "metrics"]))
    o.append(_arrow([(col[3] + bw / 2, r2 + bh), (col[3] + bw / 2, r3 + bh / 2), (col[4] - 4, r3 + bh / 2)]))
    o.append(_arrow([(col[4] + bw, r3 + bh / 2), (col[5] - 4, r3 + bh / 2)]))
    o.append(f'<text class="pnote" x="{col[2] + bw / 2}" y="{r2 + bh + 22}" text-anchor="middle">inference runs once;</text>'
             f'<text class="pnote" x="{col[2] + bw / 2}" y="{r2 + bh + 38}" text-anchor="middle">changing a metric</text>'
             f'<text class="pnote" x="{col[2] + bw / 2}" y="{r2 + bh + 54}" text-anchor="middle">never re-runs it</text>')
    o.append("</svg>")
    return "".join(o)


# An illustration, not data: twelve claims sorted by score, right (True) or wrong.
EXAMPLE = [(0.97, True), (0.93, True), (0.90, False), (0.86, True), (0.81, True), (0.77, True),
           (0.72, True), (0.66, True), (0.61, True), (0.55, True), (0.48, False), (0.41, True)]


def recall_example(claims=EXAMPLE, target: float = 0.9) -> dict:
    """The cut with the highest recall whose precision meets the target (as recording_metrics does)."""
    n_pos = sum(r for _, r in claims)
    best, tp = None, 0
    for k, (_, right) in enumerate(claims, 1):
        tp += right
        if tp / k >= target and (best is None or tp > best["tp"]):
            best = {"k": k, "tp": tp, "precision": tp / k, "recall": tp / n_pos}
    return {**best, "n_pos": n_pos, "threshold": claims[best["k"] - 1][0]}


def recall_example_svg(claims=EXAMPLE, target: float = 0.9) -> str:
    ex = recall_example(claims, target)
    W, left, cw, top = 900, 120, 60, 38
    o = [f'<svg viewBox="0 0 {W} 190" role="img" aria-label="Worked example with made-up scores: twelve claims '
         f'sorted by score; the threshold sits after the {ex["k"]}th claim, where precision is {ex["tp"]} of '
         f'{ex["k"]}, and recall is {ex["tp"]} of {ex["n_pos"]} presences.">']
    o.append(f'<text class="gsub" x="{left - 12}" y="{top + 24}" text-anchor="end">claim</text>'
             f'<text class="gsub" x="{left - 12}" y="{top - 8}" text-anchor="end">score</text>'
             f'<text class="gsub" x="{left - 12}" y="{top + 72}" text-anchor="end">precision so far</text>')
    tp = 0
    for k, (score, right) in enumerate(claims, 1):
        tp += right
        x = left + (k - 1) * cw
        o.append(f'<text class="tick" x="{x + cw / 2 - 4}" y="{top - 8}" text-anchor="middle">{score:.2f}</text>')
        o.append(f'<g class="claim {"right" if right else "wrong"}"><title>Claim {k}: score {score:.2f}, '
                 f'{"present" if right else "absent"}</title>'
                 f'<rect x="{x}" y="{top}" width="{cw - 8}" height="40" rx="5"/>'
                 f'<text x="{x + cw / 2 - 4}" y="{top + 26}" text-anchor="middle">{"right" if right else "wrong"}</text></g>')
        o.append(f'<text class="vlabel" x="{x + cw / 2 - 4}" y="{top + 72}" text-anchor="middle">{tp}/{k}</text>')
    tx = left + ex["k"] * cw - 4
    o.append(f'<line class="thr" x1="{tx}" x2="{tx}" y1="{top - 24}" y2="{top + 120}"/>')
    o.append(f'<text class="glabel" x="{tx - 8}" y="{top + 100}" text-anchor="end">threshold {ex["threshold"]:.2f}: '
             f'precision {ex["tp"]}/{ex["k"]} = {round(100 * ex["precision"])}%</text>'
             f'<text class="gsub" x="{tx - 8}" y="{top + 116}" text-anchor="end">recall: {ex["tp"]} of the '
             f'{ex["n_pos"]} presences found = {round(100 * ex["recall"])}%</text>')
    o.append(f'<text class="gsub" x="{tx + 8}" y="{top + 100}">below the threshold:</text>'
             f'<text class="gsub" x="{tx + 8}" y="{top + 116}">not claimed</text>')
    o.append("</svg>")
    return "".join(o)

"""The published report page (scripts/render_report.py), rendered from a small fixture of report CSVs.

No results/ or network needed: tests/fixtures/report holds two BirdSet sites' worth of tables.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = ROOT / "tests" / "fixtures" / "report"
RENDER = ROOT / "scripts" / "render_report.py"


def render(report: Path, out: Path, *extra: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-I", str(RENDER), "--report", str(report), "--out", str(out), *extra],
                          cwd=ROOT, capture_output=True, text=True)


@pytest.fixture(scope="module")
def page(tmp_path_factory) -> str:
    out = tmp_path_factory.mktemp("page") / "index.html"
    r = render(FIXTURE, out)
    assert r.returncode == 0, r.stderr
    return out.read_text()


def test_document_head(page):
    assert page.startswith("<!doctype html>")
    assert '<html lang="en">' in page
    assert re.search(r'<meta name="viewport" content="width=device-width, initial-scale=1">', page)
    assert '<meta name="author" content="Jordan Wolfgang Klein">' in page
    assert "<title>" in page


def test_fonts_are_self_hosted(page):
    assert "fonts.googleapis.com" not in page and "fonts.gstatic.com" not in page
    assert "assets/fonts/ibm-plex-sans-latin-400-normal.woff2" in page


def test_every_toc_link_has_a_target(page):
    nav = re.search(r'<nav class="toc".*?</nav>', page, re.S).group(0)
    hrefs = re.findall(r'href="#([^"]+)"', nav)
    ids = set(re.findall(r'\bid="([^"]+)"', page))
    assert hrefs and not [h for h in hrefs if h not in ids]


def test_page_size(page):
    assert len(page.encode()) < 3_000_000


@pytest.fixture
def nonpublic(tmp_path) -> Path:
    d = tmp_path / "report"
    shutil.copytree(FIXTURE, d)
    h = pd.read_csv(d / "headline.csv")
    h.loc[0, "evalset"] = "ourdata_scape"
    h.to_csv(d / "headline.csv", index=False)
    return d


def test_guard_refuses_nonpublic_into_docs(nonpublic):
    out = ROOT / "docs" / "_guard_test.html"
    try:
        r = render(nonpublic, out, "--allow-nonpublic")
        assert r.returncode != 0 and "ourdata_scape" in r.stderr
        assert not out.exists()
    finally:
        out.unlink(missing_ok=True)


def test_guard_needs_flag_outside_docs(nonpublic, tmp_path):
    out = tmp_path / "local.html"
    assert render(nonpublic, out).returncode != 0
    assert render(nonpublic, out, "--allow-nonpublic").returncode == 0 and out.exists()


# ---- content system (scripts/report_content.py) ------------------------------------------

sys.path.insert(0, str(ROOT / "scripts"))
import report_content as rc  # noqa: E402


def test_unknown_token_is_an_error():
    with pytest.raises(rc.TemplateError, match="unknown token"):
        rc.fill("<p>{{nope}}</p>", {"yes": 1}, {})
    with pytest.raises(rc.TemplateError, match="unknown block"):
        rc.fill("{{block:nope}}", {}, {})
    with pytest.raises(rc.TemplateError, match="unknown format"):
        rc.fill("{{x|5f}}", {"x": 0.5}, {})


def test_formats():
    facts = {"x": (0.3071, 0.4432), "a": 0.52345, "n": 1234, "s": "A & B", "p": 0.256}
    out = rc.fill("{{x|range}} {{a}} {{a|2f}} {{n}} {{s}} {{p|pct}}", facts, {})
    assert out == "0.31–0.44 0.523 0.52 1,234 A &amp; B 26%"


def test_block_and_file_render():
    blocks = {"t": "<table></table>"}
    assert rc.fill("{{block:t}}", {}, blocks) == "<table></table>"
    assert rc.fill("{{file:a/b.py}}", {}, {}) == "<code>a/b.py</code>"
    assert rc.fill("{{file:a/b.py}}", {}, {}, repo_url="https://x/r") == \
        '<a href="https://x/r/blob/main/a/b.py"><code>a/b.py</code></a>'


def test_used_facts_are_recorded():
    used = set()
    rc.fill("{{a}} {{b|2f}}", {"a": 1, "b": 2.0, "c": 3}, {}, used=used)
    assert used == {"a", "b"}


def test_rendered_page_has_no_placeholders(page):
    assert "{{" not in page and "}}" not in page.split("<script")[0]


def test_render_fails_on_unknown_token(tmp_path):
    content = tmp_path / "report"
    shutil.copytree(ROOT / "report", content)
    (content / "sections" / "conclusion.html").write_text("<p>{{no_such_fact}}</p>")
    r = render(FIXTURE, tmp_path / "x.html", "--content", str(content))
    assert r.returncode != 0 and "no_such_fact" in r.stderr and not (tmp_path / "x.html").exists()


# ---- report tables (scripts/build_report.py) ----------------------------------------------

import json  # noqa: E402

import build_report as br  # noqa: E402


def test_weakest_rule():
    sp = pd.DataFrame({
        "mode": ["clips"] * 5 + ["scape"],
        "site": ["A", "A", "B", "B", "C", "A"],
        "species": ["s1", "s2", "s3", "s4", "s5", "s1"],
        "n_pos": [50, 9, 12, 10, 30, 50],      # s2 has too few positives
        "ap_m1": [0.30, 0.01, 0.50, 0.05, 0.20, 0.0],
        "ap_m2": [0.10, 0.02, 0.05, 0.40, 0.25, 0.0],
    })
    w = br.weakest(sp, n=3, min_pos=10)
    assert list(w["species"]) == ["s5", "s1", "s4"]  # ranked by max over models: 0.25, 0.30, 0.40
    assert list(w["max_ap"]) == [0.25, 0.30, 0.40]
    assert "s2" not in set(w["species"]) and (w["n_pos"] >= 10).all()


def _pair(run: Path, evalset: str, model: str, evidence: float | None):
    d = run / evalset / model
    d.mkdir(parents=True)
    scope = {"cmAP": 0.5, "AUROC": 0.9, "top1": 0.7, "n_segments": 10, "n_classes_scored": 3, "cmAP_ci95": [0.4, 0.6]}
    (d / "summary.json").write_text(json.dumps({"evalset": evalset, "model": model, "all": scope,
                                                "covered": scope, "shared": scope}))
    t90 = {"recall": 0.4}
    if evidence is not None:
        t90 |= {"evidence_precision": evidence, "conditional_evidence_precision": evidence + 0.1}
    rl = {s: {"cmAP": 0.6, "targets": {"0.9": t90}} for s in ("all", "covered", "shared")}
    (d / "recording_level.json").write_text(json.dumps(rl))


def test_evidence_columns_for_soundscapes_only(tmp_path):
    run = tmp_path / "run"
    _pair(run, "birdset_HSN", "m", 0.9)          # clips also carry evidence in recording_level.json
    _pair(run, "birdset_HSN_scape", "m", 0.8)
    h = br.headline(br.summaries([run]))
    scape = h[(h["mode"] == "scape") & (h.site == "HSN")].iloc[0]
    clips = h[(h["mode"] == "clips") & (h.site == "HSN")].iloc[0]
    assert scape["rec_evidence90"] == 0.8 and scape["rec_evidence90_cond"] == pytest.approx(0.9)
    assert pd.isna(clips["rec_evidence90"]) and pd.isna(clips["rec_evidence90_cond"])
    all_scape = h[(h["mode"] == "scape") & (h.site == "All")].iloc[0]
    assert all_scape["rec_evidence90"] == 0.8


def test_nonpublic_evalsets_skipped(tmp_path, capsys):
    run = tmp_path / "run"
    _pair(run, "birdset_HSN_scape", "m", 0.8)
    _pair(run, "ourdata_scape", "m", 0.8)
    s = br.summaries([run])
    assert set(s["evalset"]) == {"birdset_HSN_scape"} and "ourdata_scape" in capsys.readouterr().out
    assert "ourdata_scape" in set(br.summaries([run], allow_nonpublic=True)["evalset"])

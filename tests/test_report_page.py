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

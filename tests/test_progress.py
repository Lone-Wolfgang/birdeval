"""Per-file progress: log lines every 10% when not in a terminal, thread-safe."""

from __future__ import annotations

import logging
import threading

from birdeval.models.fake import FakeModel
from birdeval.progress import Progress
from birdeval.runner import evaluate

from conftest import SPECIES


def _lines(caplog):
    return [r.getMessage() for r in caplog.records if "files (" in r.getMessage()]


def test_logs_every_ten_percent(caplog):
    caplog.set_level(logging.INFO, logger="birdeval")
    p = Progress(20, "m on e", live=False)
    for _ in range(20):
        p.update()
    lines = _lines(caplog)
    assert len(lines) == 10
    assert lines[0].startswith("m on e: 2/20 files (10%)") and lines[-1].startswith("m on e: 20/20 files (100%)")


def test_small_totals_log_each_step(caplog):
    caplog.set_level(logging.INFO, logger="birdeval")
    p = Progress(3, "x", live=False)
    for _ in range(3):
        p.update()
    assert [l.split(":")[1].split(",")[0].strip() for l in _lines(caplog)] == [
        "1/3 files (33%)", "2/3 files (67%)", "3/3 files (100%)"]


def test_updates_from_many_threads(caplog):
    caplog.set_level(logging.INFO, logger="birdeval")
    p = Progress(8000, "t", live=False)
    threads = [threading.Thread(target=lambda: [p.update() for _ in range(1000)]) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert p.n == 8000
    lines = _lines(caplog)
    assert len(lines) == 10 and "8000/8000 files (100%)" in lines[-1]


def test_runner_reports_progress_and_resume(toy, tmp_path, caplog):
    caplog.set_level(logging.INFO, logger="birdeval")
    _, tax, ev = toy
    cfg = {"output_dir": str(tmp_path / "res"), "cache_dir": str(tmp_path / "cache"), "bootstrap": 0}
    evaluate(cfg, [], [ev], run_name="p", models={"m": FakeModel("m", tax, SPECIES)})
    n = len(ev.files)
    assert any(f"{n}/{n} files (100%)" in l for l in _lines(caplog))
    # a second run with one cached file removed says it is resuming
    next((tmp_path / "cache").rglob("*.npz")).unlink()
    caplog.clear()
    evaluate(cfg, [], [ev], run_name="q", models={"m": FakeModel("m", tax, SPECIES)})
    msgs = [r.getMessage() for r in caplog.records]
    assert any(f"running inference on 1/{n} files (resuming: {n - 1} already cached)" in m for m in msgs)

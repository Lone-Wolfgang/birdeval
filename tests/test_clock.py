"""Local clock times of BirdSet soundscape files: UTC-named files and the wrap past midnight.
The HF download is stubbed: no network or BirdSet data needed."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from birdeval.evalsets import birdset

NY = "America/New_York"


def test_utc_offset_follows_daylight_saving():
    assert birdset.utc_offset_s("/x/SSW_001_20170225_010000Z.ogg", NY) == -5 * 3600  # winter: EST
    assert birdset.utc_offset_s("/x/SSW_001_20170625_010000Z.ogg", NY) == -4 * 3600  # summer: EDT
    assert birdset.utc_offset_s("/x/UHH_001_S01_20161121_150000.ogg", NY) is None  # already local


class _DS:
    def __init__(self, df):
        self.df = df

    def select_columns(self, cols):
        return _DS(self.df[cols])

    def to_pandas(self):
        return self.df


@pytest.fixture
def stub(monkeypatch):
    def use(rows):
        df = pd.DataFrame(rows, columns=["filepath", "start_time", "local_time"])
        monkeypatch.setattr(birdset, "_load", lambda config, split, cache_dir: _DS(df))
    return use


def test_file_start_is_median_of_call_clock_minus_offset(stub):
    stub([("/a/L.ogg", 10.0, "06:00:10"), ("/a/L.ogg", 70.0, "06:01:10"), ("/a/L.ogg", 30.0, "06:00:31")])
    assert birdset.file_local_start("X", NY) == {str(Path("/a/L.ogg").resolve()): 6 * 3600}


def test_utc_file_converted_and_wraps_past_midnight(stub):
    # 03:00:20 UTC on 25 Feb is 22:00:20 EST the day before; the call is 20 s into the file
    stub([("/a/SSW_001_20170225_030000Z.ogg", 20.0, "03:00:20"), ("/a/N.ogg", 0.0, None)])
    out = birdset.file_local_start("SSW", NY)
    assert out == {str(Path("/a/SSW_001_20170225_030000Z.ogg").resolve()): 22 * 3600}
    # a call 30 s into a file that started at 23:59:50 lands at 00:00:20: the start still reads 23:59:50
    stub([("/a/M.ogg", 30.0, "00:00:20")])
    assert birdset.file_local_start("X", NY) == {str(Path("/a/M.ogg").resolve()): 86400 - 10}

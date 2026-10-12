"""When and at what pitch each species calls at each BirdSet site, from the test labels.

    python -I scripts/site_calls.py [--sites HSN NBP NES PER POW SNE SSW UHH]
                                    [--out configs/reference/site_calls.csv]
                                    [--effort configs/reference/site_effort.csv]

Reads each site's soundscape `test` split (call-level events with local clock
time and a low/high frequency box), through the same eBird-code mapping as the
evaluation. No audio is decoded; file lengths come from the audio headers.

site_calls.csv   site x species: n_events, n_files (recordings with a call),
                 median low / high / centre frequency (kHz), share of calls at
                 night (20:00-05:00 local), and calls per local hour (h00..h23)
site_effort.csv  site x hour: minutes of audio recorded in that local hour.
                 Sites record on schedules, so a species' activity is
                 calls / recorded time, and hours with no recording are unknown,
                 not silent.

BirdSet's `local_time` is UTC for sites whose file names end in "Z" (PER,
SSW: SSW_001_20170225_010000Z.ogg has local_time 01:00:47); those are
converted with the site's time zone (configs/reference/sites.csv) on the
recording's date, so daylight saving is applied.

Sites whose metadata has no times or frequencies (NBP) are skipped with a note.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import yaml

from birdeval.evalsets.birdset import DAY_S, _load, event_clock
from birdeval.taxonomy import Taxonomy

SITES = ["HSN", "NBP", "NES", "PER", "POW", "SNE", "SSW", "UHH"]
NIGHT = set(range(20, 24)) | set(range(0, 5))  # fixed window; sunrise/sunset vary by site and season


def taxonomy() -> Taxonomy:
    cfg = yaml.safe_load(Path("configs/eval.yaml").read_text())["taxonomy"]
    return Taxonomy(Path("configs") / cfg["ebird_csv"], Path("configs") / cfg["synonyms_yaml"])


def effort(starts: pd.Series, lengths: pd.Series) -> np.ndarray:
    """Seconds of audio per local hour, wrapping past midnight."""
    sec = np.zeros(24)
    for s, n in zip(starts, lengths):
        t, end = s, s + n
        while t < end:
            h = int(t // 3600)
            step = min(end, (h + 1) * 3600) - t
            sec[h % 24] += step
            t += step
    return sec


def site(config: str, tax: Taxonomy, tz: str) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    ds = _load(config, "test", None)
    names = ds.features["ebird_code"].names
    mapped, _ = tax.map_many(list(names), kind="ebird")
    df = ds.select_columns(["filepath", "start_time", "end_time", "low_freq", "high_freq",
                            "local_time", "ebird_code"]).to_pandas()
    if df["local_time"].isna().all():
        print(f"{config}: no call times in the metadata, skipped")
        return None
    df["species"] = df["ebird_code"].map(lambda i: mapped[int(i)] if pd.notna(i) else None)
    df = df.dropna(subset=["species", "local_time"])
    df["clock"] = event_clock(df, tz)
    utc = df["filepath"].str.contains(r"Z\.\w+$").any()
    # each file's start on the clock: event time minus its offset into the file (median over its events)
    start = ((df["clock"] - df["start_time"]) % DAY_S).groupby(df["filepath"]).median()
    length = pd.Series({f: sf.info(f).duration for f in start.index})
    eff = effort(start, length.loc[start.index])

    df["hour"] = (df["clock"] // 3600).astype(int) % 24
    night_h = eff[sorted(NIGHT)].sum() / 3600
    ok = (df["high_freq"] > df["low_freq"]) & (df["low_freq"] >= 0)  # drop empty boxes
    df.loc[~ok, ["low_freq", "high_freq"]] = np.nan
    df["centre"] = (df["low_freq"] + df["high_freq"]) / 2
    rows = []
    for sp, g in df.groupby("species"):
        hours = np.bincount(g["hour"], minlength=24)
        rows.append({
            "site": config, "species": sp, "n_events": len(g), "n_files": g["filepath"].nunique(),
            "freq_low_khz": g["low_freq"].median() / 1000, "freq_high_khz": g["high_freq"].median() / 1000,
            "freq_centre_khz": g["centre"].median() / 1000,
            # only where the site recorded at night at all
            "night_share": g["hour"].isin(NIGHT).mean() if night_h >= 1 else np.nan,
            **{f"h{h:02d}": int(hours[h]) for h in range(24)},
        })
    eff_df = pd.DataFrame({"site": config, "hour": range(24), "minutes": eff / 60})
    rec_h = eff_df.loc[eff_df.minutes > 0, "hour"].tolist()
    print(f"{config}: {len(df):,} events, {df['species'].nunique()} species, {len(start)} files, "
          f"{'UTC->' + tz + ', ' if utc else ''}{eff.sum() / 3600:.1f} h recorded in local hours {rec_h[0]:02d}-{rec_h[-1]:02d} "
          f"({len(rec_h)} of 24 hours), {(~ok).mean():.1%} events without a frequency box")
    return pd.DataFrame(rows), eff_df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sites", nargs="+", default=SITES)
    ap.add_argument("--out", type=Path, default=Path("configs/reference/site_calls.csv"))
    ap.add_argument("--effort", type=Path, default=Path("configs/reference/site_effort.csv"))
    args = ap.parse_args()
    tax = taxonomy()
    tzs = pd.read_csv(Path("configs/reference/sites.csv")).set_index("site")["tz"]
    calls, eff = [], []
    for s in args.sites:
        res = site(s, tax, tzs[s])
        if res:
            calls.append(res[0])
            eff.append(res[1])
    pd.concat(calls).round(4).to_csv(args.out, index=False)
    pd.concat(eff).round(2).to_csv(args.effort, index=False)
    print(f"wrote {args.out} and {args.effort}")


if __name__ == "__main__":
    main()

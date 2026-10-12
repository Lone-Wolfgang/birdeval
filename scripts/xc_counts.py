"""Xeno-canto recording counts per species: a rough proxy for training data.

    export XC_API_KEY=...        # free: xeno-canto.org account -> "Your account" -> API key
    python -I scripts/xc_counts.py [--out configs/reference/xc_counts.csv]

BirdNET and Perch both train largely on xeno-canto, so a species with few
recordings there likely had little training data. Counts are as of the query
date (written to the CSV), not as of any model's training cutoff.

For every BirdSet species in results/, asks the xeno-canto API v3 for the total
number of recordings, and the number from the countries of each site where the
species occurs. Xeno-canto follows the IOC taxonomy, so when the canonical
(eBird) name finds nothing the AVONET and EltonTraits names are tried too
(configs/reference/*.csv). Species with no recordings under any name are listed.

Responses are cached under cache/xc/ (re-runs only ask what is missing; delete
it to refresh). Requests are spaced 1 s apart. The key is read from the
environment only and never written anywhere.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

API = "https://xeno-canto.org/api/3/recordings"
COUNTRIES = {  # xeno-canto country names for each site's recordings
    "HSN": ["United States"], "POW": ["United States"], "SNE": ["United States"], "SSW": ["United States"],
    "UHH": ["United States"], "PER": ["Peru"], "NES": ["Colombia", "Costa Rica"], "NBP": ["France", "Spain"],
}
CACHE = Path("cache/xc")


def count(query: str, key: str) -> int:
    """numRecordings for a xeno-canto query, cached on disk."""
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / (hashlib.sha1(query.encode()).hexdigest()[:16] + ".json")
    if f.exists():
        return int(json.loads(f.read_text())["numRecordings"])
    url = f"{API}?{urllib.parse.urlencode({'query': query, 'key': key, 'per_page': 50})}"
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                body = json.load(r)
            break
        except urllib.error.HTTPError as e:
            msg = e.read().decode(errors="replace")[:300]
            if e.code in (400, 401, 403) and "key" in msg:
                sys.exit(f"xeno-canto refused the key ({e.code}): {msg}")
            if attempt == 2:
                raise RuntimeError(f"{query}: HTTP {e.code} {msg}") from e
            time.sleep(5 * (attempt + 1))
    time.sleep(1)
    keep = {"query": query, "numRecordings": body["numRecordings"], "date": dt.date.today().isoformat()}
    f.write_text(json.dumps(keep))
    return int(body["numRecordings"])


def names(species: str, alts: dict[str, list[str]]) -> list[str]:
    return list(dict.fromkeys([species, *[a.lower() for a in alts.get(species, []) if isinstance(a, str)]]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("configs/reference/xc_counts.csv"))
    args = ap.parse_args()
    key = os.environ.get("XC_API_KEY")
    if not key:
        sys.exit("set XC_API_KEY (xeno-canto.org: log in, 'Your account', API key)")

    files = list(Path("results").glob("*/birdset_*/*/per_class_all.csv"))
    pairs = pd.concat([pd.read_csv(f, usecols=["class"]).assign(site=f.parent.parent.name.split("_")[1])
                       for f in files]).drop_duplicates()
    alts: dict[str, list[str]] = {}
    for f, col in [("configs/reference/species_traits.csv", "avonet_name"),
                   ("configs/reference/elton_traits.csv", "elton_name")]:
        if Path(f).exists():
            for sp, n in pd.read_csv(f, usecols=["species", col]).itertuples(index=False):
                alts.setdefault(sp, []).append(n)

    rows, missing = [], []
    species = sorted(pairs["class"].unique())
    for i, sp in enumerate(species, 1):
        hit, total = None, 0
        for n in names(sp, alts):
            # "celeus undatus [grammicus group]", "setophaga aestiva/petechia": first species only
            n = " ".join(n.split("/")[0].split("[")[0].split()[:2])
            total = count(f'sp:"{n}"', key)
            if total:
                hit = n
                break
        if hit is None:
            missing.append(sp)
        row = {"species": sp, "xc_name": hit, "xc_total": total}
        for site in sorted(pairs.loc[pairs["class"] == sp, "site"]):
            row[f"xc_{site}"] = (sum(count(f'sp:"{hit}" cnt:"{c}"', key) for c in COUNTRIES[site])
                                 if hit else 0)
        rows.append(row)
        if i % 25 == 0:
            print(f"  {i}/{len(species)}", flush=True)
    out = pd.DataFrame(rows)
    # one column per site is sparse; keep it long-and-simple as "count from this species' site countries"
    long = out.melt(id_vars=["species", "xc_name", "xc_total"], var_name="site", value_name="xc_site_countries")
    long = long.dropna(subset=["xc_site_countries"])
    long["site"] = long["site"].str.removeprefix("xc_")
    long["xc_site_countries"] = long["xc_site_countries"].astype(int)
    long["query_date"] = dt.date.today().isoformat()
    long.sort_values(["species", "site"]).to_csv(args.out, index=False)
    print(f"wrote {args.out}: {len(species)} species, {len(species) - len(missing)} found on xeno-canto")
    for s in missing:
        print(f"  no xeno-canto recordings under any name: {s}")


if __name__ == "__main__":
    main()

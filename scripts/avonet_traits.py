"""Species traits table from AVONET, for grouping results by habitat.

    python -I scripts/avonet_traits.py AVONET [--out configs/reference/species_traits.csv]

AVONET is ELEData.zip (or the AVONET2_eBird.xlsx inside it) from Tobias et al.
2022, Ecology Letters 25:581, CC BY 4.0:
https://figshare.com/s/b990722d72a26b5bfead (browser download only: Figshare
challenges scripted requests).

Reads the eBird-taxonomy sheet and maps each name onto birdeval's canonical
key through the taxonomy synonyms (AVONET uses the 2021 eBird taxonomy).
Then reports how many species in the BirdSet results (results/*/birdset_*/
*/per_class_all.csv) have traits. Species without a match are listed, never
dropped silently: they show up as "Unknown" in the report.
"""

from __future__ import annotations

import argparse
import io
import zipfile
from pathlib import Path

import pandas as pd

from birdeval.taxonomy import Taxonomy

SHEET = "AVONET2_eBird"
MIGRATION = {1: "Sedentary", 2: "Partial migrant", 3: "Migratory"}


def read_avonet(path: Path) -> pd.DataFrame:
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as z:
            name = next(n for n in z.namelist() if n.endswith(f"{SHEET}.xlsx") and "__MACOSX" not in n)
            return pd.read_excel(io.BytesIO(z.read(name)), sheet_name=SHEET)
    return pd.read_excel(path, sheet_name=SHEET)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("avonet", type=Path)
    ap.add_argument("--out", type=Path, default=Path("configs/reference/species_traits.csv"))
    ap.add_argument("--synonyms", type=Path, default=Path("configs/taxonomy/synonyms.yaml"))
    args = ap.parse_args()

    tax = Taxonomy(synonyms_yaml=args.synonyms)
    av = read_avonet(args.avonet)
    out = pd.DataFrame({
        "species": av["Species2"].map(tax.from_sci),
        "avonet_name": av["Species2"],
        "family": av["Family2"],
        "habitat": av["Habitat"].fillna("Unknown"),
        "migration": av["Migration"].map(MIGRATION).fillna("Unknown"),
    })
    dup = out["species"].duplicated(keep=False)
    if dup.any():
        print("several AVONET names map to one species (first kept):",
              out.loc[dup, ["species", "avonet_name"]].to_string(index=False))
    out = out.drop_duplicates("species").sort_values("species")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"wrote {len(out)} species to {args.out}")

    files = list(Path("results").glob("*/birdset_*/*/per_class_all.csv"))
    if files:
        wanted = set(pd.concat([pd.read_csv(f, usecols=["class"]) for f in files])["class"])
        missing = sorted(wanted - set(out["species"]))
        print(f"BirdSet species in results: {len(wanted)}, with traits: {len(wanted) - len(missing)}")
        for s in missing:
            print(f"  no AVONET match: {s}")


if __name__ == "__main__":
    main()

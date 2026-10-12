"""Diet, foraging stratum and nocturnality from EltonTraits 1.0.

    python -I scripts/elton_traits.py BirdFuncDat.txt AVONET [--out configs/reference/elton_traits.csv]

BirdFuncDat.txt is the bird table of EltonTraits 1.0 (Wilman et al. 2014,
Ecology 95:2027, data paper E095-178, CC0):
https://esapubs.org/archive/ecol/E095/178/BirdFuncDat.txt

EltonTraits follows the BirdLife/BirdTree taxonomy of ~2010, so each species is
tried under its own (eBird) name first, then under the BirdTree and BirdLife
names that AVONET's raw specimen table (AVONET_Raw_Data.csv in ELEData.zip,
see scripts/avonet_traits.py) pairs with its eBird name. Coverage over the
BirdSet species is printed; unmatched species are listed, never dropped.
"""

from __future__ import annotations

import argparse
import io
import zipfile
from pathlib import Path

import pandas as pd

from birdeval.taxonomy import Taxonomy, normalize_sci

STRATA = {  # EltonTraits column -> short name; percentages sum to 100 per species
    "ForStrat-watbelowsurf": "water_below",
    "ForStrat-wataroundsurf": "water_surface",
    "ForStrat-ground": "ground",
    "ForStrat-understory": "understory",
    "ForStrat-midhigh": "midhigh",
    "ForStrat-canopy": "canopy",
    "ForStrat-aerial": "aerial",
}
DIET = {"Invertebrate": "Invertebrates", "VertFishScav": "Vertebrates, fish, carrion",
        "FruiNect": "Fruit and nectar", "PlantSeed": "Plants and seeds", "Omnivore": "Omnivore"}
STRATUM_NAME = {"water_below": "Below water", "water_surface": "Water surface", "ground": "Ground",
                "understory": "Understory", "midhigh": "Mid to high", "canopy": "Canopy", "aerial": "Aerial"}


def crosswalk(avonet: Path) -> pd.DataFrame:
    """eBird name -> BirdTree/BirdLife names, from AVONET's per-specimen rows."""
    with zipfile.ZipFile(avonet) as z:
        name = next(n for n in z.namelist() if n.endswith("AVONET_Raw_Data.csv") and "__MACOSX" not in n)
        raw = pd.read_csv(io.BytesIO(z.read(name)), encoding="latin-1",
                          usecols=["Species2_eBird", "Species3_BirdTree", "Species1_BirdLife"])
    long = raw.melt(id_vars="Species2_eBird", value_name="alt")[["Species2_eBird", "alt"]].dropna()
    # most specimens first, so a lumped eBird species prefers its main BirdTree name
    return (long.groupby(["Species2_eBird", "alt"]).size().rename("n").reset_index()
            .sort_values(["Species2_eBird", "n"], ascending=[True, False]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("elton", type=Path)
    ap.add_argument("avonet", type=Path, help="ELEData.zip")
    ap.add_argument("--out", type=Path, default=Path("configs/reference/elton_traits.csv"))
    ap.add_argument("--synonyms", type=Path, default=Path("configs/taxonomy/synonyms.yaml"))
    args = ap.parse_args()

    tax = Taxonomy(synonyms_yaml=args.synonyms)
    el = pd.read_csv(args.elton, sep="\t", encoding="latin-1")
    el = el[el["Scientific"].notna()]
    el["key"] = el["Scientific"].map(normalize_sci)
    el = el.drop_duplicates("key").set_index("key")

    # candidate EltonTraits names per canonical species, in order of preference
    cands: dict[str, list[str]] = {}
    for ebird_name, g in crosswalk(args.avonet).groupby("Species2_eBird", sort=False):
        cands.setdefault(tax.from_sci(ebird_name), []).extend([normalize_sci(ebird_name), *map(normalize_sci, g["alt"])])

    files = list(Path("results").glob("*/birdset_*/*/per_class_all.csv"))
    wanted = sorted(set(pd.concat([pd.read_csv(f, usecols=["class"]) for f in files])["class"])) if files else []
    rows, missing = [], []
    for sp in sorted(set(cands) | set(wanted)):
        hit = next((c for c in [sp, *cands.get(sp, [])] if c in el.index), None)
        if hit is None:
            if sp in wanted:
                missing.append(sp)
            continue
        r = el.loc[hit]
        strata = {v: float(r[k]) for k, v in STRATA.items()}
        rows.append({
            "species": sp, "elton_name": r["Scientific"],
            "diet": DIET.get(r["Diet-5Cat"], r["Diet-5Cat"]),
            # ties go to the lower stratum (dict order)
            "stratum": STRATUM_NAME[max(strata, key=strata.get)],
            **{f"strat_{k}": v for k, v in strata.items()},
            "nocturnal": int(r["Nocturnal"]) == 1,
        })
    out = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"wrote {len(out)} species to {args.out}")
    if wanted:
        print(f"BirdSet species in results: {len(wanted)}, with EltonTraits: {len(wanted) - len(missing)}")
        for s in missing:
            print(f"  no EltonTraits match: {s}")


if __name__ == "__main__":
    main()

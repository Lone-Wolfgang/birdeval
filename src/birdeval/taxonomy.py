"""Mapping every label system onto one canonical species key.

Canonical key = normalised scientific name, e.g. "turdus merula".

- BirdNET 2.4 / 3.0 labels look like "Turdus merula_Eurasian Blackbird".
- Perch V2 labels are scientific names (plus some non-bird classes).
- BirdSet labels are eBird species codes ("eurbla"), mapped through the eBird
  taxonomy CSV.

Taxonomies drift between versions (splits, lumps, genus changes), so a
user-maintained synonyms file maps old names onto the name used in your eval
sets. Unmapped labels are reported, never silently dropped.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd
import yaml

log = logging.getLogger(__name__)

_WS = re.compile(r"\s+")


def normalize_sci(name: str) -> str:
    return _WS.sub(" ", name.strip().lower())


class Taxonomy:
    def __init__(
        self,
        ebird_csv: str | Path | None = None,
        synonyms_yaml: str | Path | None = None,
    ) -> None:
        self.code_to_sci: dict[str, str] = {}
        if ebird_csv:
            df = pd.read_csv(ebird_csv)
            cols = {c.upper(): c for c in df.columns}
            code_col = cols.get("SPECIES_CODE")
            sci_col = cols.get("SCI_NAME") or cols.get("SCIENTIFIC_NAME")
            if not code_col or not sci_col:
                raise ValueError(
                    f"{ebird_csv}: expected SPECIES_CODE and SCI_NAME columns, got {list(df.columns)}"
                )
            self.code_to_sci = {
                str(c).strip(): normalize_sci(str(s)) for c, s in zip(df[code_col], df[sci_col])
            }
        self.synonyms: dict[str, str] = {}
        if synonyms_yaml and Path(synonyms_yaml).exists():
            raw = yaml.safe_load(Path(synonyms_yaml).read_text()) or {}
            self.synonyms = {normalize_sci(k): normalize_sci(v) for k, v in raw.items()}

    def _resolve(self, sci: str) -> str:
        sci = normalize_sci(sci)
        return self.synonyms.get(sci, sci)

    def from_sci(self, name: str) -> str:
        return self._resolve(name)

    def from_birdnet(self, label: str) -> str:
        """'Turdus merula_Eurasian Blackbird' -> 'turdus merula'."""
        return self._resolve(label.split("_", 1)[0])

    def from_ebird(self, code: str) -> str | None:
        sci = self.code_to_sci.get(code.strip())
        return self._resolve(sci) if sci else None

    def map_many(self, labels: list[str], kind: str) -> tuple[list[str | None], list[str]]:
        """Map a list of labels; returns (mapped, unmapped_originals)."""
        fn = {"sci": self.from_sci, "birdnet": self.from_birdnet, "ebird": self.from_ebird}[kind]
        mapped = [fn(l) for l in labels]
        unmapped = [l for l, m in zip(labels, mapped) if m is None]
        if unmapped:
            log.warning("%d %s labels could not be mapped, e.g. %s", len(unmapped), kind, unmapped[:5])
        return mapped, unmapped

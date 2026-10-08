# birdeval

One harness, any model × any eval set. Models and eval sets live in a single
registry (`configs/eval.yaml`); pick combinations at the command line.

```
birdeval run -m birdnet_2.4 birdnet_3.0 perch_v2 -e birdset_HSN birdset_POW
```

## Setup

```bash
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -e ".[models,perch,birdset,dev]"  # BirdNET 2.4/3.0, Perch V2 (TF >= 2.20; not on Intel macOS), BirdSet
pytest                                            # runs without weights or downloads
birdeval check --audio some_bird.wav              # load every model, run it, show top species
```

`birdeval check` downloads weights on first use (BirdNET 3.0 is the slow one,
about 6 min the first time), runs each model on a generated 12 s clip and a
2 s clip (shorter than any window), and validates shapes and score ranges.
Any `--audio` clips get a top-3 species list. A model that fails is reported
without stopping the rest, and the exit code is nonzero.

Then put an eBird/Clements taxonomy CSV at `configs/taxonomy/ebird_taxonomy.csv`
(needs `SPECIES_CODE` and `SCI_NAME` columns). BirdSet labels are eBird codes,
so this is required.

**First real run:** use the smoke set to shake out data-loading issues before
committing hours of inference:

```bash
birdeval run -m birdnet_2.4 -e birdset_POW_smoke
```

## How it works

```
EvalSet loader ──> segments (file, start, end, labels) ─┐
                                                        ├─> align ─> metrics ─> results/
ModelAdapter ──> window scores ──> cache (.npz) ────────┘
```

- **One canonical label space**: normalised scientific names. BirdNET's
  `"Sci_Common"` labels, Perch's scientific names and BirdSet's eBird codes are
  all mapped onto it (`taxonomy.py`). Name changes between taxonomy versions go
  in `configs/taxonomy/synonyms.yaml`.
- **All three models go through the official `birdnet` package** (≥ 1.1),
  which loads BirdNET 2.4, BirdNET 3.0 and Perch V2 behind one `predict()`.
  The adapter asks for full score vectors (no top-k / threshold cut-off),
  restricted to the eval set's classes.
- **Raw window scores are cached before alignment.** Changing the alignment
  rule or adding a metric never re-runs inference. The cache key hashes every
  model parameter that changes output, so changing `overlap_s` or a model
  version starts a fresh cache.
- **Alignment** (`align.py`): a segment's score per class is the max over model
  windows overlapping it by ≥ `min_overlap_s`. Same rule for every model.
- **Metrics** (`metrics.py`): cmAP, AUROC, top-1 (BirdSet's set), plus a
  file-level bootstrap CI on cmAP (segments within a file are correlated, so
  resampling segments would give CIs that are too narrow).

### Three class scopes, on purpose

Every result is reported three ways:

| scope   | classes scored                              | use for                         |
|---------|---------------------------------------------|---------------------------------|
| all     | every eval class (unpredictable ones = 0)   | "how useful is this out of the box" |
| covered | classes this model can predict              | per-model diagnostics           |
| shared  | classes **every model in the run** can predict | **comparing models**         |

`uncovered_classes.txt` per model lists what fell through. Check it after the
first run: entries there that the model *should* know are taxonomy mismatches
to add to `synonyms.yaml`.

## Outputs

```
results/<run>/summary.csv                     one row per model × evalset × scope
results/<run>/<evalset>/<model>/summary.json  metrics, CI, coverage, model + alignment config
results/<run>/<evalset>/<model>/per_class_<scope>.csv
results/<run>/<evalset>/<model>/uncovered_classes.txt
```

## Adding things

- **A model**: subclass `ModelAdapter` (`labels`, `to_canonical`,
  `predict_files`), register it in `models/__init__.py:ADAPTERS`, add a config
  entry. A fine-tuned head or a different backend of an existing model is
  usually just a new config entry.
- **An eval set**: write a loader returning an `EvalSet`, register it in
  `evalsets/__init__.py:LOADERS`. For your own data, `strong_labels` already
  handles event annotations (file, start, end, label → 5 s segments) and
  `segments_csv` handles pre-segmented labels. See the commented `mine:` entry
  in the config.

## Known gaps / things to verify

- **BirdSet loader is untested against the live dataset.** Column names and
  the `filepath` behaviour are from BirdSet's documented layout; check
  `_COLS` in `evalsets/birdset.py` on the first run. BirdSet uses a HF loading
  script, hence `datasets<4`.
- **The `birdnet` adapter runs with real weights** (birdnet 1.1.1, TF 2.21,
  onnxruntime 1.30, Apple Silicon). All three models put *Turdus merula* first
  on a blackbird recording. This covers loading and inference, not BirdSet
  data loading.
- **Perch V2 scores saturate under sigmoid.** Perch's logits sit well above 0
  even for absent species (about 7–9 for background species and 12–14 for the
  true species on the blackbird clip), so sigmoid pushes almost everything
  toward 1. birdnet clips logits at ±15 before the sigmoid, so very confident
  windows tie. Ranking metrics are unaffected until logits reach 15. Before
  trusting Perch numbers, decide whether to score it with raw logits or softmax.
- **BirdNET 3.0 is a developer preview.** Labels and scores will change between
  releases; the package version is recorded in every `summary.json`.
- **Window overlap matters.** With max-pooling, overlapping windows near a
  segment boundary can carry a call into the neighbouring segment (the test
  suite hit this). Each model's `overlap_s` and the global `min_overlap_s`
  together set how much; keep them fixed across models you compare, and
  consider a small sweep once things run.
- No geo-model (location/week) filtering yet. BirdSet soundscapes have
  locations, so this is a natural next experiment.

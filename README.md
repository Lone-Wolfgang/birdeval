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
2 s clip (shorter than any window), and checks shapes and that scores are finite.
Any `--audio` clips get a top-3 species list. A model that fails is reported
without stopping the rest, and the exit code is nonzero.

Then put an eBird taxonomy CSV at `configs/taxonomy/ebird_taxonomy.csv`
(needs `SPECIES_CODE` and `SCI_NAME`/`SCIENTIFIC_NAME` columns). BirdSet labels
are eBird codes, so this is required. The eBird API serves it without a key:

```bash
curl -s "https://api.ebird.org/v2/ref/taxonomy/ebird?fmt=csv" -o configs/taxonomy/ebird_taxonomy.csv
```

**First real run:** use a smoke set to shake out data-loading issues before
committing hours of inference:

```bash
birdeval run -m birdnet_2.4 -e birdset_POW_scape_smoke   # 3 soundscapes
```

### BirdSet: clips or soundscapes

BirdSet's `test_5s` split is not 5 s slices of longer recordings. Every row
is a separate, pre-cut 5 s audio file. So there are two modes, both scored on
BirdSet's official 5 s labels:

| eval set | audio the model hears | use for |
|---|---|---|
| `birdset_<SITE>` | the published 5 s clips | comparing with BirdSet papers |
| `birdset_<SITE>_scape` | the full soundscapes; each clip's labels sit at its offset | recording-level results; matches deployment |

In soundscape mode, any time no clip covers is not reviewed. Both modes
download only BirdSet's test material.

`scripts/birdset_labels.py` reports label density per site and cross-checks
the soundscape labels against labels rebuilt from BirdSet's call-level
annotations, using metadata only. The cross-check gives exact segment
agreement of 96–100% at seven sites (100% at POW). **NBP is not a
soundscape set:** its 539 "recordings" are about 5 s each, and only 52% of
segments agree. Recording lengths differ by site: POW 300 s, HSN 600 s,
UHH 60–540 s, NES/PER/SNE/SSW 1 h.

## How it works

```
EvalSet loader ──> segments (file, start, end, labels) ─────────┐
                                                               ├─> align ─────> segment metrics
ModelAdapter ──> raw logits ──> cache (.npz) ──> scorer ───────┤
                                                               └─> aggregate ─> recording-level metrics
```

- **One canonical label space**: normalised scientific names. BirdNET's
  `"Sci_Common"` labels, Perch's scientific names and BirdSet's eBird codes are
  all mapped onto it (`taxonomy.py`). Name changes between taxonomy versions go
  in `configs/taxonomy/synonyms.yaml`.
- **All three models go through the official `birdnet` package** (≥ 1.1),
  which loads BirdNET 2.4, BirdNET 3.0 and Perch V2 behind one `predict()`.
  The adapter asks for full score vectors (no top-k / threshold cut-off),
  restricted to the eval set's classes.
- **Raw logits are cached before alignment.** Changing the alignment rule,
  the scorer or a metric never re-runs inference. The cache key hashes every
  model parameter that changes output (including `score_space`), so changing
  `overlap_s` or a model version starts a fresh cache. BirdNET 3.0 exports
  only probabilities, so its logits are reconstructed
  (`score_space: logit_from_prob`).
- **Scorer** (`scoring.py`, `scoring.method`): `logit` (default) or an
  unclipped `sigmoid`, applied to cached logits. Ranking metrics are identical
  for both. Softmax is deliberately not offered: in soundscapes several
  species call at once, and softmax makes them compete.
- **Alignment** (`align.py`): a segment's score per class is the max over model
  windows overlapping it by ≥ `min_overlap_s`. Same rule for every model.
- **Metrics** (`metrics.py`): cmAP, AUROC, top-1 (BirdSet's set), plus a
  bootstrap CI on cmAP that resamples whole groups (site, else source
  recording, else file). Segments within a recording are correlated, so
  resampling segments would give CIs that are too narrow.

### Three class scopes, on purpose

Every result is reported three ways:

| scope   | classes scored                              | use for                         |
|---------|---------------------------------------------|---------------------------------|
| all     | every eval class (unpredictable ones score lowest) | "how useful is this out of the box" |
| covered | classes this model can predict              | per-model diagnostics           |
| shared  | classes **every model in the run** can predict | **comparing models**         |

`uncovered_classes.txt` per model lists what fell through. Check it after the
first run: entries there that the model *should* know are taxonomy mismatches
to add to `synonyms.yaml`.

## Recording-level results

Deployment claims "species X is present in recording R" and backs each claim
with a clip someone can listen to. Precision matters more than recall. So
alongside the segment metrics, every run reports two recording-level layers
(`recording_level` in the config; on by default):

- **Presence (headline numbers):** each (unit, species) pair gets a
  confidence: the max window score in the unit. A unit is a whole file, or
  `unit_s`-long chunks to match a deployment's recording length. The pair
  is present if any segment in the unit is.
  - *rec cmAP*: per-species AP across units.
  - *Recall at precision ≥ target* (defaults 0.8 / 0.9 / 0.95): the lowest threshold whose
    pooled precision meets the target, and the recall it gets.
  - *List precision*: at that threshold, the fraction of claimed species that
    are really there. Pooled, and averaged per unit.
  - *By duration*: the same, per duration bin, at the global threshold. With
    max-pooling, longer recordings give absent species more chances at one
    high window.
  - *Per species*, at each pooled global threshold: precision and recall,
    showing which species a single deployed threshold serves badly.
  - Bootstrap CIs resample sites (if the eval set has a `site` column), else
    source recordings. Chunks of one recording (`unit_s`) and BirdSet clips
    of one soundscape always move together.
- **Evidence:** the claim's evidence clip is its max-scoring window.
  *Evidence precision* is how often that clip contains the species.
  *Conditional evidence precision* counts only claims that are right at the
  unit level, which separates "wrong species" from "right species, wrong clip".

`fp_review.csv` lists the highest-scoring wrong claims with their clip times.
It's the list for an expert to listen to: some "false positives" will be
missed labels, which tells us where the precision ceiling really is.

### Labels: three states

Ground truth is 1 (present), 0 (absent) or −1 (not reviewed). Unreviewed cells
are skipped by every metric, at both levels, so a real but unlabelled call
the model finds is neither a hit nor a false positive. For your own data, set
`reviewed_species` and/or `reviewed_ranges` on `strong_labels` /
`segments_csv` (see the commented `mine:` entry in the config). The defaults
treat everything unlabelled as absent, which gives the same numbers as
before. BirdSet is exhaustively annotated, so it always uses the defaults.

> Results on our own data assume exhaustive labelling. If labelling was not
> exhaustive, precision is a lower bound.

## Outputs

```
results/<run>/summary.csv                     one row per model × evalset × scope (segment + rec_* columns)
results/<run>/<evalset>/<model>/summary.json  segment metrics, CI, coverage, model / scoring / alignment config
results/<run>/<evalset>/<model>/per_class_<scope>.csv
results/<run>/<evalset>/<model>/uncovered_classes.txt
results/<run>/<evalset>/<model>/recording_level.json  presence + evidence metrics per scope, by duration, CIs
results/<run>/<evalset>/<model>/presence.csv          unit, species, presence_conf, features, evidence window, truth
results/<run>/<evalset>/<model>/detections.csv        file, start, end, species, confidence, unit,
                                                      presence_correct, evidence_correct
results/<run>/<evalset>/<model>/fp_review.csv         top wrong claims to listen to, by score
results/<run>/<evalset>/<model>/per_species_recording_<scope>.csv  per-species AP, precision / recall
                                                      at each global threshold
```

`detections.csv` uses the threshold that meets `operating_target` (0.9) on
the `all` scope. It is empty, with a warning, if no threshold reaches it.

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

- **BirdSet loaders are checked live on POW only.** Clip-name parsing and
  soundscape lookup hold for all 8 sites' metadata. Uncovered soundscape
  time has only been measured at POW, where there is none. BirdSet uses a HF
  loading script, hence `datasets<4`.
- **Unreviewed time makes absent species unknown for the whole unit.** This
  follows the three-state rule. If a site's soundscapes have short
  unreviewed tails (e.g. dropped partial clips), every absent species in
  those recordings drops out of the recording-level metrics. Check
  `uncovered_s` in the eval set metadata before trusting a site's
  recording-level numbers.
- **The `birdnet` adapter runs with real weights** (birdnet 1.1.1, TF 2.21,
  onnxruntime 1.30, Apple Silicon). All three models put *Turdus merula* first
  on a blackbird recording. This covers loading and inference, not BirdSet
  data loading.
- **Perch V2 logits are high for everything** (about 7–9 for absent species,
  12–14 for present ones on the blackbird clip). birdnet's own sigmoid clips
  at ±15 and caused ties, so the cache now holds raw logits and the scorer
  stage decides what metrics see. Raw Perch scores are not probabilities, and
  a fixed threshold won't transfer between models. Compare models on the
  threshold-free metrics or on recall at a target precision, never at a
  shared raw threshold.
- **Training-data overlap.** Before comparing models on BirdSet, check each
  model's documentation for whether any BirdSet test sites were in its
  training data. A model that trained on a site will look better there than
  it will in deployment.
- **BirdNET 3.0 is a developer preview.** Labels and scores will change between
  releases; the package version is recorded in every `summary.json`.
- **Window overlap matters.** With max-pooling, overlapping windows near a
  segment boundary can carry a call into the neighbouring segment (the test
  suite hit this). Each model's `overlap_s` and the global `min_overlap_s`
  together set how much; keep them fixed across models you compare, and
  consider a small sweep once things run.
- No geo-model (location/week) filtering yet. BirdSet soundscapes have
  locations, so this is a natural next experiment.

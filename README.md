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

### Checking against published figures

```bash
python -I scripts/compare_published.py results/taxfix_v1
```

This compares clip-mode rows (`birdset_<SITE>`, scope `all`, which is
BirdSet's protocol) with the per-site values in
`configs/reference/published.yaml`, and writes
`results/published_comparison.csv`. It and `build_report.py` read each
pair's `summary.json`, so they also work on a run stopped before it wrote
`summary.csv`.
- **Perch V2** is checked against the three Perch 2.0 variants in the Perch 2.0
  paper (App. C, Table 4: HSN and SNE, no POW). A value more than `--tol`
  (0.02) outside their range is flagged.
- **BirdNET** has no published per-site figures. BirdSet excluded it for
  possible test-data leakage, so BirdNET results on BirdSet may be optimistic.
- Context rows (Perch 1.0 and two BirdSet baselines) give scale only.

Taxonomy mismatches pull the `all` scope down: an uncovered class scores
AUROC 0.5. Before `synonyms.yaml` had its current entries, Perch V2 was
0.01–0.02 below the published cmAP and AUROC on HSN and SNE, and its
`covered` scope fell inside the published range.

### Report

```bash
python -I scripts/build_report.py results/taxfix_v1 results/full_*   # tables -> results/report/*.csv
python -I scripts/render_report.py                                  # page   -> results/report/index.html
```

This writes the report's tables to `results/report/`. Clips and soundscape
runs are both read; if two runs hold the same (mode, site, model), the later
one wins.
- `headline.csv`: mode × site × model, plus an `All` row per mode. That row
  is the mean of the per-site values over the sites every model has
  finished.
- `published.csv`: the published-figure check.
- `species.csv`: one row per mode × site × species. Each row has the
  positives count, the species' AVONET traits, the site biome, and each
  model's AP, AUROC and recall at 90% precision. Soundscape rows add
  recording-level AP.
- `groups.csv`: mean per-species AP per mode × scope × grouping.
  - Scope is all sites, or one site.
  - Grouping is AVONET primary habitat, site biome, or prevalence.
  - The prevalence bins (1–9, 10–49, 50+ labelled segments) were fixed in
    advance.

The habitat and family columns need `configs/reference/species_traits.csv`,
from AVONET (Tobias et al. 2022, *Ecology Letters* 25:581, CC BY 4.0).
To build it:
1. Download `ELEData.zip` in a browser from
   https://figshare.com/s/b990722d72a26b5bfead. Figshare blocks scripted
   downloads.
2. Run `python -I scripts/avonet_traits.py ELEData.zip`. This needs the
   `report` extra.

Site names, WWF biomes and time zones are in `configs/reference/sites.csv`.

#### Species information

Each source has its own script. Each script writes a CSV to
`configs/reference/` and prints how many BirdSet species it covers, listing
any it misses. `build_report.py` joins whatever CSVs exist.

| Script | Source | Adds |
|---|---|---|
| `avonet_traits.py ELEData.zip` | AVONET (CC BY 4.0) | habitat, habitat density, migration, body mass, wing length, trophic niche, primary lifestyle |
| `elton_traits.py BirdFuncDat.txt ELEData.zip` | [EltonTraits 1.0](https://esapubs.org/archive/ecol/E095/178/BirdFuncDat.txt) (Wilman et al. 2014, CC0) | diet category, foraging stratum, nocturnal flag |
| `site_calls.py` | BirdSet's call-level test labels | for each site and species: median call frequency band, calls per local hour, and share of calls at night; `site_effort.csv` holds recorded minutes per hour |
| `xc_counts.py` | xeno-canto API v3 | recordings per species, in total and from the site's country |

Common names and eBird codes come from the eBird taxonomy CSV. The page
links each species to xeno-canto, eBird and the Macaulay Library.

Notes on the sources:
- **EltonTraits:** it uses a ~2010 taxonomy, so names are matched through
  AVONET's per-specimen eBird↔BirdTree↔BirdLife names. Its foraging strata
  are noisy for some species. For example, Apapane is listed as 100% ground,
  and its "aerial" means high above the vegetation, so swallows score low.
- **BirdSet times:** `local_time` is UTC for PER and SSW, whose file names end
  in `Z`. `site_calls.py` converts it using the site's time zone. Most sites
  record only around dawn; SSW records all day and UHH 05–23 h. NBP has no
  times or frequency boxes.
- **xeno-canto:** needs a free account. Get the API key under "Your account"
  and `export XC_API_KEY=...`. The key is read from the environment only.
  Responses are cached in `cache/xc/`. Counts are as of the query date, not
  as of any model's training cutoff.

#### By hour of day

```bash
python -I scripts/hourly.py        # -> results/report/hourly.csv, then re-render
```

Bird activity and model performance by local hour, pooled over species. It
reads the prediction cache and **never runs inference**: a site is skipped,
with a message, unless every model has every soundscape file cached. It
rebuilds the same segment scores as `birdeval run`; the all-hours AP and
AUROC match each run's `per_class_all.csv`. NBP has no clock times.

Each 5 s segment gets the local hour of its midpoint. The file start comes
from BirdSet's call times (`birdset.file_local_start`), with UTC-named files
converted. Per site, hour and model it reports:
- Activity: share of segments with a bird, species per segment, species
  heard.
- AUROC, the main measure: it does not depend on how common calls are, so
  busy and quiet hours compare fairly. cmAP is there too, but a quiet hour
  lowers AP even for an unchanged model.
- Recall, precision and false alarms per hour of audio at fixed thresholds.
  Each species' threshold gives 90% segment precision over the site's whole
  test set and is set before the split by hour.

Only hours with at least 30 minutes of reviewed audio are kept. `All` shows,
for each hour, the mean over sites of the change from that site's all-day
value, so it is not driven by which sites record at which hours; its activity
columns pool the sites. 95% intervals resample recordings within the
hour (200 draws).

A group's unit is a (site, species) pair, and its 95% interval comes from
resampling those pairs. Groups use only sites that every model has finished,
so the models are compared on the same pairs.

The page has a static summary and published check. Below them is an
explorer with three controls:
- Mode: clips or soundscapes.
- Dataset: all sites, or one site.
- View: overview, groups (by habitat, site biome, prevalence or a species
  trait), species, or hours.

Every table sorts when you click a column heading. The species table also
has:
- A metric picker.
- A model-vs-model difference column.
- A search box.
- A filter for species with fewer than 10 positives. It is on by default.

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
- **Alignment** (`align.py`): a segment's score per class is the max over the
  model windows that count towards it. A window counts if more than half of
  it lies inside (`alignment.rule: half_window`), measured on its own length.
  It's one geometric rule for every model, with nothing to tune, and no
  window counts towards two segments. Recording-level units use the same
  rule. The old global `min_overlap_s` let Perch's 5 s windows score calls
  in the neighbouring segment (POW soundscapes: cmAP 0.471 vs 0.567 on
  clips), and no single threshold suited both 3 s and 5 s windows.
- **Metrics** (`metrics.py`): cmAP, AUROC, top-1 (BirdSet's set), plus a
  bootstrap CI on cmAP that resamples whole groups (site, else source
  recording, else file). Segments within a recording are correlated, so
  resampling segments would give CIs that are too narrow.
  Each draw reweights presorted scores (a group drawn twice counts twice)
  instead of copying and re-sorting, and the draws run on `workers`
  processes (config; default min(8, cores)). The numbers equal the
  copy-and-resort computation and don't depend on `workers`.

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
`synonyms.yaml` also maps retired eBird codes that BirdSet still uses (keys
like `runwre1`). With the current entries, every model covers every BirdSet
test class except four for BirdNET 2.4, which lacks those species: Campylorhynchus
capistratus and two Saucerottia hummingbirds (NES), and Curruca cantillans (NBP).

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
  *Evidence precision* is how often that clip contains the species. It is
  checked against the annotated calls when the eval set has them (BirdSet,
  `strong_labels`): the clip must overlap a call of that species by ≥ 0.5 s.
  Being in a labelled 5 s segment isn't enough.
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

### Watching a run

In a terminal, each model × eval set shows a progress bar (files done, rate,
ETA). With output redirected to a file, the same information is logged every
10%. Inference is cached file by file, so an interrupted run resumes where it
stopped: just run the same command again.

## Outputs

```
results/<run>/summary.csv                     one row per model × evalset × scope (segment + rec_* columns); rewritten after each eval set
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
- **Units are trimmed to reviewed time.** Unreviewed stretches (e.g. soundscape
  audio no labelled clip covers) are left out of a recording's unit rather
  than making every absent species unknown. A presence claim is about the
  reviewed part of the recording. `recording_level.json` records how much
  was trimmed.
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
- **Segment-level results depend on the alignment rule.** The half-window rule
  replaced a global overlap threshold on 2026-10-08. Earlier segment-level
  numbers aren't comparable; recording-level numbers barely moved.
- No geo-model (location/week) filtering yet. BirdSet soundscapes have
  locations, so this is a natural next experiment.

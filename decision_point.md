# birdeval: decision point on BirdSet data for recording-level evaluation

*Written 2026-10-08, after Phases 1–4 of `spec_recording_level_evel.md` were built. This is meant to be read on its own, without access to the repo.*

## Summary

The harness can now evaluate at the recording level: is species X present in recording R, and does the clip cited as evidence actually contain it? Phases 1–4 of the spec are built and tested. The spec's last check is a real run on BirdSet's POW smoke set, and it can't give meaningful recording-level results as the code stands. The BirdSet split we load, `test_5s`, isn't 5-second slices of longer recordings. It's 4,560 separate 5-second audio files, so every "recording" would be one 5 s clip.

We need to decide how to give the harness real recordings from BirdSet. My recommendation is a **soundscape mode**: run models on BirdSet's full 5-minute recordings, using BirdSet's official 5 s labels placed back on each recording's timeline. The current clip mode stays, for comparison with published BirdSet numbers. Option D below is a cheaper alternative worth weighing.

---

## 1. Background

**birdeval** is one harness for any model × any eval set. Models and eval sets live in a config file, and a run picks combinations. The longer-term goal is a dashboard where users swap models (base or fine-tuned) and evaluate them on public and private datasets.

How it works today:

```
eval set loader ──> labelled segments (file, start, end, species) ─┐
                                                                   ├─> align ─────> segment metrics
model ──> raw logits per window ──> cache ──> scorer ──────────────┤
                                                                   └─> aggregate ─> recording-level metrics
```

- **Models:** BirdNET 2.4, BirdNET 3.0 and Perch V2, all loaded through the official `birdnet` Python package (1.1.1).
- **Eval sets:** BirdSet test sets from Hugging Face (8 sites: HSN, NBP, NES, PER, POW, SNE, SSW, UHH). Our own expert-labelled data, about 25k labels, goes in through `strong_labels` / `segments_csv`.
- **Deployment context that drives the design:** passive recorders capture whole soundscapes. The output is a detection table (`start, end, species, confidence`) whose rows a person can listen to. **Precision matters more than recall.**

## 2. What has been done

### 2.1 Every target model loads and runs

- Set up an isolated environment (Python 3.12, birdnet 1.1.1, TensorFlow 2.21, onnxruntime 1.30) on an Apple M4 Pro.
- Added a `birdeval check` command. It loads each model, runs it on generated clips and on any real recordings given, checks the output shapes and values, and prints the top species.
- On a Eurasian Blackbird recording, all three models rank *Turdus merula* first.
- Bugs fixed along the way:
  - Perch V2 crashed on every call, because of a birdnet setting default.
  - The birdnet version recorded in results was always "unknown".
  - A test failed on macOS because of a path quirk.

### 2.2 Spec Phases 1–4

Each phase is its own commit, and the full test suite (38 tests) passes after each one.

| Phase | What it does | Key evidence |
|---|---|---|
| 1. Raw logits + scorer | The cache stores raw logits, and a scorer step (`logit` or `sigmoid`) is applied after the cache, so changing it never re-runs a model. BirdNET 3.0 only outputs probabilities, so its logits are reconstructed. | On a real clip, sigmoid of the stored logits matches the old scores to within 9e-8 (BirdNET 2.4 and Perch). |
| 2. Three-state labels | Present / absent / not reviewed. Metrics skip anything not reviewed. | A test confirms that with default settings, cmAP, AUROC, top-1 and the bootstrap interval equal the old code exactly. |
| 3. Recording-level presence | Each (unit, species) pair gets a confidence: the best window score in the unit. Reports cmAP across recordings, recall at a target precision (default 0.9), list precision, results by recording length, and bootstrap intervals. | Recall-at-precision is tested against hand-computed answers, including tied scores. |
| 4. Evidence + review export | Checks whether the cited clip contains the species. Writes `detections.csv` (the deployment table plus truth columns) and `fp_review.csv` (the top wrong claims, for an expert to listen to). | The spec's planted case is tested: the right species is claimed, but the cited clip contains a different species. |

Phase 5 (a calibrated presence score) is deliberately not started. The spec says to begin it only after looking at a real BirdSet run.

### 2.3 Why Phase 1 mattered

On the blackbird clip, Perch V2's raw scores were about 7–9 for absent species and 12–14 for the species actually there. birdnet's own sigmoid caps scores at ±15, so Perch's most confident windows were tying with each other, and after the sigmoid everything read 1.00. BirdNET 2.4 had the opposite problem: about 65k scores fell below −15, all tied at the bottom.

Raw logits remove both kinds of tie. They also make one rule explicit: **raw scores from different models are not comparable.** Models should be compared on threshold-free metrics, or on recall at a fixed precision target. Never compare them at a shared raw threshold.

### 2.4 Places where I interpreted or went beyond the spec

These are worth a second look:

1. **Scores for classes a model can't predict** used to be fixed at 0.0. In logit space 0.0 means 50%, which would let those classes win top-1. They now default to the lowest possible score.
2. **Evidence is "unknown"** if the cited clip overlaps any unreviewed segment and no labelled call. The spec said "all unreviewed". I used "any" because the missed call could be in the unreviewed part. This matches how recording-level truth is defined.
3. **Results by recording length** use the global operating threshold rather than picking a new threshold for each length bin. Picking per bin would hide the effect we're looking for: longer recordings give absent species more chances to score one high window.
4. **`fp_review.csv`** lists the top wrong claims whether or not they pass the threshold, so there's a listening list even when the 0.9 precision target is never reached.
5. **Files that appear only in `reviewed_ranges`** (reviewed, no birds) are now included as negatives. Before, they were silently dropped.
6. **No before/after comparison on real data.** A BirdSet run has never been done, so the spec's requirement that "segment numbers match the pre-change run" can only be checked on synthetic data, which the tests do.

## 3. The problem: what BirdSet's `test_5s` split actually is

I checked the POW metadata files directly. They are small parquet files on the Hugging Face repo's `data` branch.

| | `test_5s` (what we load today) | `test` |
|---|---|---|
| Rows | 4,560 | 16,052 |
| A row is | a **separate 5 s audio file**, e.g. `Recording_1_Segment_03_000_005.ogg`, with start 0 and end 5 | one labelled call (start, end, eBird code) inside a full recording |
| Audio | its own shard, about 200 MB | full recordings, about 440 MB |
| Labels | multi-label eBird codes per 5 s clip (BirdSet's official labels) | call-level ("strong") labels |

- **POW:** 76 recordings, each exactly 300 s (60 clips each), 48 species. 97% of clips have at least one label.
- **The clip filename encodes where it came from.** `Recording_1_Segment_03_000_005.ogg` is seconds 0–5 of `Recording_1_Segment_03.ogg`.

**Why this blocks recording-level evaluation:** the loader treats each audio file as a recording. With `test_5s`, every recording is a single 5 s clip. "Is species X anywhere in this recording?" becomes "is it in this 5 s clip?", which is just the segment-level task again. The recording-length breakdown and the evidence check (where in the recording is the call?) lose their meaning. A smoke run would technically succeed and the numbers would look plausible, but they wouldn't measure what the spec is for.

Segment-level evaluation on `test_5s` is still valid. It's BirdSet's published protocol.

## 4. Options

### A. Keep clips only (status quo)

- **Pro:** no work, and comparable to published BirdSet numbers.
- **Con:** recording-level results on BirdSet are meaningless. They'd only be usable on our own data.

### B. Soundscape mode with official labels (recommended)

- Load the 76 full recordings from the `test` split, and run models on continuous audio.
- Labels come from `test_5s`: each clip's labels go onto its recording's timeline at the offset in its filename. That gives the same 5 s grid and the same official labels, just on full recordings.
- **Pro:** this matches deployment, since recorders capture continuous audio and model windows can span 5 s boundaries as they do in the field. A unit is a real 300 s recording, so recording-level results and evidence times are meaningful. The labels are exactly BirdSet's.
- **Con:**
  - Segment numbers will differ slightly from BirdSet's published clip protocol, because windows can now cross clip boundaries.
  - It's an extra download (about 440 MB per site for POW; other sites differ).
  - The loader needs a filename-to-offset mapping that we must check holds across all 8 sites.

### C. Soundscape mode with our own segmentation of call-level labels

- Same audio as B, but labels are built from the call-level `test` labels using our existing event-to-segment rule. A segment counts as positive if a call overlaps it by at least 0.5 s.
- **Pro:** reuses the existing `strong_labels` code, and the call-level labels are richer (exact call times).
- **Con:** our overlap rule may not match how BirdSet derived its 5 s labels, so results stop being on BirdSet's official labels. It's best used as a **cross-check** of B: if B's and C's labels mostly agree, the mapping is right.

### D. Group clips into virtual recordings (cheapest)

- Keep loading `test_5s` clips and running models on each 5 s clip. Add a `recording` column parsed from the filename, and let a unit span several clip files (the 60 clips of one recording).
- **Pro:** no new audio download. Segment numbers stay identical to BirdSet's published protocol, and recording-level results become meaningful.
- **Con:**
  - Models never see continuous audio, so calls cut at clip boundaries are scored in pieces. Window placement within 5 s clips is also odd: BirdNET's 3 s windows give a partial last window per clip.
  - That differs from deployment.
  - Aggregation currently assumes one audio file per unit, so it would need generalising. That's moderate work.

### Recommendation

**Build B, keep A, and use C once as a one-off check of B's label mapping.** A is still useful for comparison with BirdSet papers. The deciding argument is that the spec is about *deployment* claims on continuous recordings, and only B and C evaluate models the way they'll be used.

D is a reasonable alternative if download size or segment-level comparability with published BirdSet numbers matters more than matching deployment. The difference between B and D at recording level is probably small, but that's an empirical question we haven't measured.

## 5. Things to know about POW whichever option is chosen

These affect how much the first results can tell us:

1. **All 76 recordings are exactly 300 s.** With whole-file units, the recording-length breakdown has a single bin, so it's useless. Setting `unit_s: 60` gives 380 one-minute units across one length bin. To actually see a length effect, run several `unit_s` values (e.g. 30, 60, 150, 300) and compare.
2. **76 recordings is a small sample.** Bootstrap intervals on recording-level numbers will be wide. Treat POW as the smoke test, and pool sites, or use HSN/SSW etc., for real conclusions.
3. **POW is dense.** 97% of clips are labelled, and the common species will be present in most recordings. Recording-level precision for common species will look easy. Per-species results and `fp_review.csv` matter more than the pooled headline.
4. **POW is one site (Powdermill).** Bootstrapping by site isn't possible, so the harness resamples by file.
5. **Training-data overlap.** Before comparing models on BirdSet, check whether any model trained on recordings from BirdSet's test sites. This is noted in the README, but nobody has checked the actual model documentation yet.

## 6. Prerequisites for any BirdSet run (not done yet)

- Install `datasets<4`. BirdSet uses a Hugging Face loading script, and `datasets` 4.x removed support for those.
- Get the eBird taxonomy CSV. The public eBird API serves it without a key, with `SPECIES_CODE` and `SCIENTIFIC_NAME` columns that the harness already reads.
- BirdSet's loading script and the column names in our loader have never been run against the live dataset. The first run will confirm them.

## 7. Questions to settle

1. **B (continuous audio, matches deployment) or D (clips grouped, matches BirdSet's protocol, no new download)?** Or B now and D later, to measure how much the choice matters?
2. **What unit lengths should we evaluate?** Does the deployment produce a fixed recording length (e.g. 1 min, 5 min, 1 h)? That decides `unit_s` and the length bins.
3. **Is 0.9 the right precision target, and is one global target enough?** Phase 5 adds per-species thresholds. Should anything move earlier?
4. **Which sites after POW?** I'd suggest one dense site and one sparse site, to see how prevalence affects recording-level precision.
5. **Our own 25k-label set:** was labelling exhaustive, per species or per time range? This decides whether we set `reviewed_species` and/or `reviewed_ranges`, and whether its precision numbers are a lower bound.
6. **Interpretation choices in section 2.4:** are any of them wrong for how results will be used?

## 8. What happens after the decision

1. Build the chosen BirdSet mode, with tests, as one commit.
2. Install the prerequisites and run the spec's final check: `birdeval run -m birdnet_2.4 perch_v2 -e birdset_POW_smoke`. Confirm that `presence.csv`, `detections.csv`, `fp_review.csv` and `recording_level.json` are written and make sense.
3. Review that first real run (score distributions per model, `fp_review.csv`, coverage gaps from taxonomy mismatches) before starting Phase 5 (calibration), as the spec says.

# Spec: recording-level, precision-first evaluation for birdeval

**For:** Claude Code, working in the `birdeval` repo.
**Status:** ready to implement in phases. Each phase ends with green tests and a commit.

---

## 0. Before you start

The repo has moved on from the original scaffold (e.g. there is now a `birdeval check`
command). **Read the current code first** and adapt the module and function names in this
spec to what actually exists. Where this spec conflicts with existing behaviour, keep the
existing segment-level results working and add the new behaviour alongside.

Rules for the whole job:

- One phase per commit. Run `pytest` after each phase; all tests must pass.
- Don't change existing segment-level metric values (cmAP / AUROC / top-1 on `all`,
  `covered`, `shared`). Phase 1 is the only exception, and only because ties at the top go
  away (see its acceptance test).
- Every new config key needs a default that reproduces today's behaviour, so old configs
  still run.
- No new heavy dependencies. numpy / pandas / scikit-learn are enough.
- If a check in this spec turns out wrong for the installed `birdnet` version (e.g. the
  meaning of `apply_sigmoid=False` for a model), stop and report it rather than working
  around it silently.

---

## 1. Context: what we're evaluating and why

- **Deployment:** passive recorders capturing whole soundscapes.
- **What deployment claims:** "species X is present in recording R." The goal is evidence of
  which species are present in a specific recording.
- **What deployment outputs:** a detection table with `start, end, species, confidence`. Those
  rows are the *evidence* a person can listen to.
- **Priority:** precision over recall.

So the eval must report two layers:

1. **Presence (the headline numbers):** is each species reported for a recording really
   there?
2. **Evidence:** does the clip cited for a reported species actually contain that species?

Known facts that drive the design:

- Perch V2 logits are high for absent species (about 7–9) as well as present ones (12–14).
  Sigmoid pushes nearly everything toward 1, and `birdnet` clips logits at ±15 before the
  sigmoid, which creates ties among the most confident windows. **Store raw logits.**
- Softmax isn't used for scoring: in soundscapes several species call at once and softmax
  makes them compete. (BirdNET 3.0 exports can't produce it anyway.)
- Our own 25k expert-labelled set may not be exhaustively labelled. **Labels need a third
  state: not reviewed.**
- With max-pooling, longer recordings give absent species more chances at one high score.
  **Report recording-level results by duration.**

---

## 2. Phase 1: raw logits in the cache, and a scorer stage

### 2.1 Cache raw logits

In the `birdnet` adapter:

- BirdNET 2.4 and Perch V2: call `predict` with `apply_sigmoid=False`,
  `apply_softmax=False`, `default_confidence_threshold=None`. Keep the current `top_k` /
  `custom_species_list` logic.
- BirdNET 3.0: its exports apply the sigmoid inside the graph and expose only
  probabilities. Store `logit(clip(p, 1e-7, 1 - 1e-7))`, and record in `version_info()` that
  the scores are reconstructed (`"score_space": "logit_from_prob"`).
- Add `"score_space": "logit"` (or `"logit_from_prob"`) to `version_info()` so the cache key
  changes and old sigmoid caches are never mixed with logit caches.
- **Verify** with `birdnet check` or a short script: for 2.4 and Perch, sigmoid of the stored
  value should equal the old score (to float tolerance, below the old clip limit).

### 2.2 Scorer stage

Add `scoring.py`, applied after the cache is read and before alignment / aggregation:

```yaml
scoring:
  method: logit      # logit | sigmoid    (calibrated is added in Phase 5)
```

- `logit`: identity.
- `sigmoid`: `1 / (1 + exp(-x))` with no clipping (use a numerically stable form).
- The scorer is a pure function on `WindowScores`. It must not trigger re-inference.

### 2.3 Tests

- `logit` and `sigmoid` give identical cmAP and AUROC on the synthetic eval set (ranking
  invariance).
- Reconstructing logits from probabilities round-trips within tolerance for p in
  [1e-6, 1 − 1e-6].
- The cache key differs between the old and new `score_space`.

---

## 3. Phase 2: three-state labels

### 3.1 Representation

The ground-truth matrix becomes `int8` with three values:

| value | meaning |
|---|---|
| 1 | present |
| 0 | absent (reviewed, not present) |
| −1 | not reviewed (unknown) |

- `EvalSet.target_matrix()` returns this matrix. Keep a helper that returns the old binary
  form (unknown → 0) where existing code needs it, but metrics must use the masked form.
- **Default:** everything not labelled present is 0 (absent). With default settings, results
  are identical to today.

### 3.2 Loader options (`strong_labels`, `segments_csv`)

```yaml
reviewed_species: all          # all | [list of species] — species outside the list are -1
reviewed_ranges: null          # optional CSV: file, start, end — time outside is -1 for all species
```

BirdSet loader: always `reviewed_species: all`, no ranges (exhaustively annotated by
design).

### 3.3 Metrics

- Segment-level metrics (`metrics.compute`, `bootstrap_cmap`) ignore −1 cells, per class:
  for class c, use only the rows where `y[:, c] != -1`.
- top-1: a row is used only if it has at least one positive and its top-scoring class is not
  −1 in that row.

### 3.4 Tests

- With defaults, segment-level metrics match the pre-change values exactly.
- Marking a species as not reviewed removes it from `n_classes_scored`. Marking a time range
  as not reviewed removes those rows from the class computations.
- A synthetic case where an unlabeled (not reviewed) true call gets a high score: precision
  is lower when that cell counts as absent than when it's −1.

---

## 4. Phase 3: recording-level presence

### 4.1 Units

A **unit** is what a presence claim is about. By default it's the whole file. Optionally it
can be fixed-length chunks, which lets you match deployment recording length and get more
units from datasets with few, long files:

```yaml
recording_level:
  unit_s: null            # null = whole file; or e.g. 60, 300
  duration_bins_s: [60, 300, 900, 3600]
  min_overlap_s: 1.0      # reuses alignment's rule for window/segment overlap
```

### 4.2 Ground truth per (unit, species)

- present (1): any segment in the unit is 1 for that species
- absent (0): all segments in the unit are 0
- unknown (−1): no 1s and at least one −1. Excluded from all recording-level metrics.

### 4.3 Aggregation (`aggregate.py`)

Work from the scored `WindowScores` (not the aligned segment matrix), so evidence times are
the model's real windows. For each (unit, species) compute:

| feature | definition |
|---|---|
| `max_score` | max window score in the unit |
| `top3_mean` | mean of the 3 highest window scores |
| `n_above` | windows with score ≥ `tau` (config, on the scorer's scale) |
| `n_windows` | windows in the unit |
| `duration_s` | unit duration |
| `evidence_start`, `evidence_end` | the window with `max_score` |

Baseline presence score = `max_score`. Phase 5 replaces it with a calibrated probability.

### 4.4 Metrics (`recording_metrics.py`)

All computed on units with known ground truth, for each class scope (`all` / `covered` /
`shared`, same definitions as the segment level):

- **Per-species AP** across units, and its mean (recording-level cmAP).
- **Recall at precision ≥ target** (default target 0.9, configurable, list allowed): pooled
  over all (unit, species) pairs, take the highest-recall threshold whose precision meets
  the target. Report the threshold, the precision achieved and the recall. If no threshold
  reaches the target, report recall 0 and say so.
- **List precision at the operating threshold:** of all (unit, species) claims above the
  threshold, the fraction that are present. Report pooled (micro) and as the mean
  per-unit value.
- **By duration:** the same three numbers per `duration_bins_s` bin.
- **Uncertainty:** bootstrap CI over units for recall-at-precision and list precision (if a
  `site` column exists, resample sites instead).

### 4.5 Outputs

```
results/<run>/<evalset>/<model>/recording_level.json     metrics above, per scope
results/<run>/<evalset>/<model>/presence.csv             unit, species, presence_conf, n_above,
                                                         evidence_start, evidence_end, truth
```

Add recording-level headline columns to `results/<run>/summary.csv` (e.g. `rec_cmAP`,
`rec_recall_at_p90`, `rec_list_precision`), or write a separate `summary_recording.csv` if
that's cleaner in the current code.

### 4.6 Tests

- Synthetic: a species present in one segment makes the unit present; all-zero makes it
  absent; any −1 without a 1 makes it unknown and excluded.
- Recall at precision: on a hand-built score list, the chosen threshold and recall match the
  hand-computed answer.
- `unit_s: 60` on a 300 s file produces 5 units with the correct segment membership.

---

## 5. Phase 4: evidence precision and the review export

### 5.1 Definition

For each (unit, species) claimed present at the operating threshold:

- The **evidence window** is the window with `max_score`.
- The evidence is **correct** if it overlaps a segment labelled 1 for that species by at
  least `min_overlap_s`.
- If the overlapping segments are all −1 for that species, the evidence is **unknown** and
  excluded.

Report:

- **Evidence precision:** correct / (correct + incorrect), over all claims.
- **Conditional evidence precision:** the same, restricted to claims that are correct at the
  unit level. This separates "wrong species claimed" from "right species, wrong clip".

### 5.2 Outputs

```
results/<run>/<evalset>/<model>/detections.csv
    file, start, end, species, confidence, unit, presence_correct, evidence_correct
```

This is the PI's table format plus truth columns.

```
results/<run>/<evalset>/<model>/fp_review.csv
```

The top N (config, default 200) highest-scoring presence claims that are wrong, with their
evidence clip times, sorted by score. This is the list for the expert to listen to. It tells
us how many "false positives" are missed labels and so where the precision ceiling really is.

### 5.3 Tests

- Planted case: a species correctly present in the unit, but its highest window is in a
  stretch where only another species is labelled. Unit-level correct; evidence incorrect.
- `fp_review.csv` is sorted by score and contains only incorrect claims.

---

## 6. Phase 5: calibrated presence confidence

Start this phase only after Phases 1–4 are green and you've looked at a real BirdSet run.

### 6.1 Splits (no leakage)

- Grouped K-fold (default K = 5) over **files**, or over **sites** if a `site` column exists
  (`group_by: file | site`). Use `sklearn.model_selection.GroupKFold`.
- The calibrator is fit on K−1 folds and predicts the held-out fold, so every unit gets an
  out-of-fold calibrated score and all units can still be evaluated.
- A test asserts that no file (or site) is in both the train and the predict part of any
  fold.

### 6.2 Model

```yaml
scoring:
  method: calibrated
  calibration:
    group_by: file            # file | site
    k_folds: 5
    features: [max_score, top3_mean, n_above, log_n_windows]
    species_effects: true     # one-hot species term with L2 penalty (partial pooling)
    C: 1.0
```

- `sklearn.linear_model.LogisticRegression` on (unit, species) rows with known truth.
- `species_effects: true` adds a one-hot species indicator, so common species get their own
  offset and rare species fall back to the pooled fit.
- Output `presence_conf` in [0, 1] replaces `max_score` in `presence.csv`, `detections.csv`
  and the recording-level metrics.
- Leave hooks (feature names in config, no code yet) for `geo_prior` and
  `other_model_score`. Those are the next experiments, not part of this job.

### 6.3 Per-species thresholds for a target precision

Within each training fold, pick a threshold per species that reaches the target precision
(fall back to the pooled threshold for species with fewer than `min_pos` positives, default
10). Apply it to the held-out fold. Report list precision and recall with these thresholds,
next to the global-threshold numbers from Phase 3.

### 6.4 Calibration report

- Reliability curve: 10 equal-count bins of `presence_conf` with the observed present rate per
  bin. Write it to `calibration.csv`.
- Expected calibration error (ECE).
- Both computed out-of-fold.

### 6.5 Tests

- Synthetic data where the true probability is known: after calibration, ECE is below a
  loose bound (e.g. 0.05 with enough rows).
- No group appears in both parts of any fold.
- With `species_effects: false` the model has exactly `len(features)` coefficients plus
  an intercept.

---

## 7. Config: everything new in one place

```yaml
scoring:
  method: logit                 # logit | sigmoid | calibrated
  calibration: {group_by: file, k_folds: 5, features: [max_score, top3_mean, n_above, log_n_windows],
                species_effects: true, C: 1.0, min_pos: 10}

recording_level:
  enabled: true
  unit_s: null
  duration_bins_s: [60, 300, 900, 3600]
  tau: 0.0                      # n_above threshold, on the scorer's scale (0.0 logit = 0.5 prob)
  precision_targets: [0.9]
  fp_review_n: 200

# per eval set (strong_labels / segments_csv only):
#   reviewed_species: all
#   reviewed_ranges: null
```

---

## 8. Docs

Update the README:

- Replace the "Perch V2 scores saturate" gap with how it's now handled (raw logits; scorer
  stage).
- Add a short "Recording-level results" section explaining presence vs evidence, list
  precision and recall at precision, and the new output files.
- Add a "Labels: three states" note, and this footnote for results on our own data until we
  know whether labelling was exhaustive: *"Assumes exhaustive labelling. If not, precision is
  a lower bound."*
- Add a "Training-data overlap" caveat: before comparing models on BirdSet, check each model's
  documentation for whether any BirdSet test sites were in its training data.

---

## 9. Done when

- [ ] `pytest` passes, including all new tests listed above.
- [ ] `birdeval run -m birdnet_2.4 perch_v2 -e birdset_POW_smoke` writes `presence.csv`,
      `detections.csv`, `fp_review.csv`, `recording_level.json` (and `calibration.csv` with
      `scoring.method: calibrated`).
- [ ] Segment-level numbers with default settings match the pre-change run (except tie-breaking
      changes from Phase 1, which should be noted in the commit message).
- [ ] README updated as in section 8.
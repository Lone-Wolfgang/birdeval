# birdeval: decision point on matching model windows to labelled segments

*Written 2026-10-08, after the soundscape-mode build and the first real BirdSet run. This is meant to be read on its own, without access to the repo. The previous decision write-up (BirdSet clips vs soundscapes) is in git history at commit `badc4c3`.*

## Summary

The harness now evaluates models on BirdSet's full soundscape recordings. The first real run (BirdNET 2.4, BirdNET 3.0 and Perch V2 on the 76 POW recordings) is done. Recording-level results look sound.

Segment-level results exposed a problem with how model windows are matched to 5 s labelled segments. With the current rule, a single global `min_overlap_s = 1.0`, Perch V2's 5 s windows score calls in the neighbouring segment. Its segment cmAP drops from 0.567 on clips to 0.471 on soundscapes. Raising the threshold to 2.6 s fixes Perch (0.552) but costs BirdNET 2.4 about 0.015 cmAP, because its 3 s windows are excluded more often.

**The current lean is option 1: a per-model overlap rule.** It reuses the cached inference, needs no new model runs, and leaves recording-level results essentially unchanged. The open questions are how to choose each model's value without tuning on the test set, and how to report it.

---

## 1. Background

**birdeval** is one harness for any model × any eval set, heading towards a dashboard for comparing base and fine-tuned bird-sound models on public and private data. Deployment is passive recorders capturing soundscapes. Precision matters more than recall.

The pipeline:

```
eval set ──> labelled 5 s segments ─────────────────────┐
                                                        ├─> align (max-pool windows → segments) ──> segment metrics
model ──> raw logits per window ──> cache ──> scorer ───┤
                                                        └─> aggregate (per recording) ──────────────> recording-level metrics
```

- **Models:**
  - BirdNET 2.4: 3 s windows, 1.5 s hop.
  - BirdNET 3.0: 3 s windows, 1.5 s hop; a developer preview.
  - Perch V2: 5 s windows, 2.5 s hop.
- **Labels:** BirdSet's official 5 s segment labels.
- **Alignment rule today:** a segment's score for a species is the **max** over model windows that overlap the segment by at least `min_overlap_s` (one global value, default 1.0 s). This is the only part of the pipeline the decision affects.

## 2. What was built and tested since the last decision

The previous decision chose **soundscape mode**: run models on BirdSet's full recordings, and place the official 5 s clip labels at each clip's offset. All requested items were built, each committed with tests (51 tests passing):

| Commit | What |
|---|---|
| `badc4c3` | **Soundscape mode** (`birdset_<SITE>_scape` eval sets). Time not covered by any labelled clip is marked "not reviewed". Bootstrap resamples whole recordings (site, else source recording, else file), including when recordings are chunked into shorter units. Precision targets 0.8 / 0.9 / 0.95, with 0.9 as the operating point. Per-species precision and recall at each global threshold. Tests pin a finite floor score for classes a model can't predict. |
| `91a602e` | **Label report script**, from metadata only: label density for all 8 BirdSet sites, and the cross-check of soundscape labels against labels rebuilt from call-level annotations. |
| `ad66105` | README for all of the above. |
| `fc86ba3` | **Crash-safe inference.** Each file is cached as soon as it's scored, and writes are atomic. An interrupted run resumes where it stopped. |
| `30068e8` | **Performance fix.** Recording-level metrics took about 15 min per model on 4,560 clips because of a quadratic loop. They now take seconds, with byte-identical outputs. |

Along the way we found that the original BirdSet loader would have downloaded about 16 GB of training audio per site. Both modes now use BirdSet's test-only download.

### 2.1 Verification results

**Soundscape labels are right.** Labels rebuilt from BirdSet's call-level annotations (a call counts if it overlaps a segment by ≥ 0.5 s) agree exactly with the official labels on:

| | POW | HSN | SSW | NES | UHH | SNE | PER | NBP |
|---|---|---|---|---|---|---|---|---|
| Exact segment agreement | **100%** | 99.4% | 99.4% | 99.2% | 98.9% | 98.4% | 96.2% | 52.0% |

NBP isn't a soundscape set: it's 539 "recordings" of about 5 s each. On POW, soundscape mode and clip mode carry identical labels (12,911 labelled cells each).

**Label density per site:**

| Site | Recordings | Length | Hours | Species | Clips labelled | Species per recording |
|---|---|---|---|---|---|---|
| POW | 76 | 300 s | 6.3 | 48 | 97% | 9.0 |
| HSN | 100 | 600 s | 16.7 | 21 | 45% | 3.6 |
| UHH | 635 | 60–540 s | 50.9 | 27 | 71% | 4.0 |
| NES | 34 | 1 h | 34 | 89 | 39% | 10.7 |
| PER | 21 | 1 h | 21 | 132 | 90% | 37.4 |
| SNE | 33 | 1 h | 33 | 56 | 69% | 9.9 |
| SSW | 285 | 1 h | 285 | 81 | 32% | 7.5 |
| NBP | 539 | ≈5 s | 0.6 | 51 | 100% | 1.7 |

**The new code reproduces the old results.** On all 4,560 POW clips:
- BirdNET 2.4's segment cmAP, AUROC and top-1 are identical.
- Perch's cmAP differs by +0.00015, from tie-breaking (the old pipeline's float32 sigmoid compressed scores near 1).
- Raw scores agree to within 1.8e-7.
- Confidence intervals are wider now, because the bootstrap resamples whole recordings instead of individual 5 s clips. That's the more honest interval.
- The 3-clip smoke set was too small to score any species in either version.

**Crash safety was tested for real.** A run was killed with `kill -9` after its first recording was cached. It left no orphaned processes or temp files, and the restart scored only the remaining 2 of 3 recordings.

### 2.2 First real run: POW soundscapes (76 recordings × 300 s)

Shared scope. 3,572 (recording, species) pairs, 684 present (19%).

| | BirdNET 2.4 | BirdNET 3.0 | Perch V2 |
|---|---|---|---|
| Segment cmAP (95% CI), current alignment | 0.588 (0.566–0.664) | 0.674 (0.649–0.719) | 0.471 (0.461–0.539) |
| Recording cmAP | 0.767 | 0.875 | 0.793 |
| Recall at 90% precision (95% CI) | 0.785 (0.76–0.81) | 0.825 (0.76–0.86) | 0.556 (0.50–0.61) |
| Recall at 95% precision | 0.757 | 0.734 | 0.480 |
| Evidence precision at 90% | 0.847 | 0.864 | 0.870 |
| Evidence precision, right-species claims only | 0.940 | 0.959 | 0.966 |

Reading these:
- **Recording-level results are reliable and barely depend on alignment.** They take the max over the whole recording, so window-to-segment matching hardly matters (see §3).
- **Perch ranks species well but suffers from one shared threshold.** Its recording cmAP (0.793) beats BirdNET 2.4's, but its recall at fixed precision is much lower. Its raw scores sit at different levels for different species, so one global cutoff serves it badly. Per-species calibration (spec Phase 5) is the planned fix.
- **Evidence clips are reliable.** When the species claim is right, the cited clip contains it 94–97% of the time.
- **Only one site, 76 recordings,** and BirdNET 3.0 is a developer preview.

## 3. The decision: matching windows to segments

### 3.1 The problem

With max-pooling, a window that overlaps a segment contributes its score to it, even if most of the window lies in the next segment.

- **Perch:** 5 s windows at a 2.5 s hop. Every 5 s segment is overlapped by its own window plus two neighbours, each by 2.5 s. With `min_overlap_s = 1.0`, both neighbours count, so a call in segment *k* also scores in segments *k−1* and *k+1*.
- **BirdNET:** 3 s windows at a 1.5 s hop. Windows straddling a boundary overlap the neighbour by 1.5–2.5 s, so there's some bleed too, but less.
- **Clip mode hides all of this,** because each clip is a separate file and windows can't cross clip boundaries.

### 3.2 Evidence: `min_overlap_s` sweep from cache

A parallel session ran this; I checked the outputs.

| `min_overlap_s` | Perch, soundscapes | Perch, clips | BirdNET 2.4, soundscapes | BirdNET 2.4, clips |
|---|---|---|---|---|
| 1.0 (current) | cmAP 0.471, top-1 0.839 | 0.567 | cmAP 0.588, top-1 0.935 | 0.594 |
| 2.6 | cmAP **0.552**, top-1 **0.936** | 0.551 | cmAP 0.574, top-1 0.979 | 0.579 |

These are segment-level results on the classes each model can predict.

- **At 2.6 s, only Perch's own centred window counts,** and Perch on soundscapes matches Perch on clips (0.552 vs 0.551). The drop was an alignment artifact, not the model.
- **2.6 s hurts BirdNET's cmAP.** A 3 s window can overlap a 5 s segment by at most 3 s, so 2.6 s excludes most of its windows and leaves fewer chances to catch a call. BirdNET's top-1 *rises* (0.935 → 0.979), so bleed was affecting BirdNET too.
- **Recording-level cmAP moves by at most 0.002.**
- **BirdNET 3.0 wasn't in the sweep.**

So no single global value is neutral across window lengths.

### 3.3 Options

1. **Per-model overlap rule (current lean).** Each model gets its own `min_overlap_s`; the global value stays as the default.
   - Pros: reuses the cache with no new inference; quick; keeps each model's natural windowing.
   - Cons: segment results now depend on a per-model setting that must be chosen and justified.
2. **Align windows to the 5 s grid.** For example, Perch with no window overlap (hop = 5 s); BirdNET would need a hop that tiles 5 s segments.
   - Pros: cleanest comparisons; nothing to tune.
   - Cons: needs new inference; changes what the model sees (fewer windows, so less chance to catch calls at boundaries); doesn't map cleanly onto BirdNET's 3 s windows.
3. **Keep one global value and document the bias.**
   - Pros: no work.
   - Cons: segment-level comparisons between Perch and BirdNET stay biased in BirdNET's favour.

### 3.4 Points to settle for option 1

1. **How to choose each model's value without tuning on the test set.** Picking 2.6 s for Perch because it gave the best POW numbers is tuning on test data. Better to fix a rule from window geometry *before* looking at results. Candidates:
   - **"Most of the window lies inside the segment":** overlap > window length / 2. Perch: > 2.5 s. BirdNET (3 s windows): > 1.5 s.
   - **"Only windows whose centre falls in the segment":** a rule about window position rather than an overlap amount. Each window then contributes to exactly one segment, which removes double counting.
   - **Check either rule:** each model's soundscape result should land close to its clip result, as Perch did at 2.6 s.
   - Neither rule has been run yet. BirdNET at 1.5 s or with the centre rule is untested, and so is BirdNET 3.0 (which also accepts variable segment lengths).
2. **Config shape.** For example `models.<name>.min_overlap_s`, overriding `alignment.min_overlap_s`. The value should be recorded in each `summary.json` and should probably not change the cache key, since alignment happens after the cache.
3. **What else uses `min_overlap_s`.** Recording-level unit membership and the evidence-correctness check reuse the alignment value. Should they follow the per-model value too? Recording-level results barely move, but evidence correctness does depend on which windows overlap labelled calls.
4. **Reporting.** Each run should record the rule used per model, and the README should explain that segment-level comparisons are "model + its alignment rule". Should we also keep reporting a common global setting, so results stay comparable with BirdSet's published protocol?
5. **Is a centre-based rule better than an overlap threshold?** It's simpler to explain, has nothing to tune, and gives each window to exactly one segment. The cost is a little sensitivity at segment edges.

## 4. Other open items, not part of this decision

- **Phase 5 (per-species calibration).** It's now motivated by Perch's shared-threshold problem.
  - `n_above` is useless as a calibration feature for Perch: with `tau = 0`, every one of its windows counts as "above", because its raw scores are always positive. `tau` needs to be set per model.
- **Training-data overlap:** nobody has yet checked whether any model trained on recordings from BirdSet's test sites.
- **Uncovered time at the other sites:** measured only at POW (none). Some sites' last clips are partial (e.g. HSN, NES, SNE, UHH). Under the three-state rule, any unreviewed tail makes every absent species "unknown" for that whole recording, which would remove those negatives from recording-level metrics. This needs checking before trusting those sites.
- **More sites:**
  - one sparse site (e.g. SSW: 32% of clips labelled, 1 h recordings, but 285 h of audio);
  - one long-recording site, to see whether recording length affects precision;
  - PER is very dense (37 species per recording).
- **`fp_review.csv` is ready for expert listening.** BirdNET 2.4's top "false positive" on POW is a Black-capped Chickadee scored highly in 13 windows of one recording, a likely missed label.
- **Housekeeping:** two scratch git worktrees (`baseline` at the old commit, `crashsafe`) and the merged branch `crash-safe-cache` can be removed.

## 5. Questions to discuss

1. Option 1 with a fixed geometric rule (overlap > half the window, or the window's centre in the segment), or per-model values chosen another way?
2. Should evidence correctness and recording-level unit membership follow the per-model rule?
3. Do we keep a common-setting result alongside, for comparison with BirdSet's published protocol?
4. After this: Phase 5 calibration next, or more sites first?

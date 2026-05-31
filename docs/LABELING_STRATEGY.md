# Labeling Strategy

## Purpose

This document defines how each of the ~631k clips is labeled as **meaningful**, **not_meaningful**, or **unknown** for the audio detection model. It records the decisions that are settled and explicitly marks the part that is still open.

The previous labeling approach (BirdNET confidence >= 0.3 as the meaningful/not-meaningful oracle) was rejected. See `DIAGNOSIS_LABELS.md`: a listening study found that ~93% of the model's apparent false positives were real meaningful sound that BirdNET missed at threshold 0.3. BirdNET at 0.3 is not a trustworthy oracle for "meaningful."

## Definition of "meaningful"

A clip is **meaningful** if it contains bird sound, other animal sound, or human-activity sound (speech, footsteps, vehicles, gunshots, chainsaws). Insects are treated as meaningful: they are animal sound, and they frequently co-occur with bird sound, so including them prevents real bird clips from being filtered out before the downstream classifier.

A clip is **not_meaningful** if it contains only background / ambient sound (rain, wind, silence) with no bird, animal, or human-activity content.

## Label scheme

Two columns are added to the working dataframe:

- `meaningful` — one of `meaningful`, `not_meaningful`, `unknown`
- `meaningful_source` — how the clip was labeled (`human_activity`, `birdnet_species`, ..., or `unlabeled`)

Every clip starts as `unknown` / `unlabeled`. Each labeling step "rescues" clips it is confident about into a definite label with a recorded source. Clips that no step can confidently label remain `unknown` and are excluded from training rather than guessed.

## Carve-out 1: Human activity (gold standard)

During field deployment, 63 simulation events (Human Presence on/off trail, Vehicle, Gunshot, Chainsaw) were staged at known times near specific recorders. These events are recorded in `event_table.csv` and joined onto the clip catalog as the `Sim Type` column.

Any clip inside a simulation event window is labeled **meaningful**, source **human_activity**. This is the most trustworthy label in the dataset — it is ground truth from controlled field events, independent of any model.

Result: **10,871 clips** labeled meaningful via human_activity.

## BirdNET named-species audit

Before trusting BirdNET's species output as a label, its reliability was audited by listening. BirdNET reports a named species for ~108,855 clips, at confidence levels ranging from below 0.3 up to above 0.9.

A stratified sample of 100 named-species clips (20 per confidence bin across five bins: 0.0-0.3, 0.3-0.5, 0.5-0.7, 0.7-0.9, 0.9-1.0) was listened to and tagged bird / no-bird / unsure.

**Result: 100 / 100 clips contained bird sound, across all confidence bins.** Confidence tracked the *clarity* of the bird sound (higher confidence = clearer / closer), not whether a bird was present. A low BirdNET confidence does not mean "probably not a bird" — it means a faint or distant bird.

Conclusion: when BirdNET names a species, the clip contains a bird, regardless of confidence level. The audit found 0 errors in 100 sampled clips. "Named species" is therefore a trustworthy meaningful signal at any confidence.

## Carve-out 2: BirdNET named species

Any clip with a named species (`species != "[]"`) is labeled **meaningful**, source **birdnet_species** — at any confidence level.

Note: this is broader than the old confidence >= 0.3 rule, which discarded ~23,000 named-species clips below 0.3. The audit confirmed those low-confidence named-species clips are real birds, so they are now correctly included.

Clips that already carry the `human_activity` label keep that source (gold standard wins); only still-unknown named-species clips receive the `birdnet_species` source. Both are meaningful — this only affects source bookkeeping.

Result: **108,069 clips** labeled meaningful via birdnet_species (the ~786 named-species clips that overlapped with human-activity events retained the human_activity source).

## Current state

| Label | Source | Count |
|---|---|---|
| meaningful | birdnet_species | 108,069 |
| meaningful | human_activity | 10,871 |
| unknown | unlabeled | 512,377 |
| **Total** | | **631,317** |

Confident meaningful labels so far: **118,940**. Remaining unknown: **512,377** (~81%).

## Open problem: the unknown pool (in progress)

The 512,377 unknown clips are those where BirdNET named no species and no simulation event occurred. This pool is **not** pure background. The listening study (see `DIAGNOSIS_LABELS.md`) established that clips of this kind frequently contain real birds and insects that BirdNET missed.

No oracle is available for this pool — BirdNET has already returned "nothing" for all of them. Tools potentially available to attack it:

- The trained CNN model itself, which flags many of these as meaningful and was ~93% correct on the clips it flagged in the listening study.
- BirdNET's raw confidence signal (even where no species was named).
- Targeted listening on samples.
- Leaving the pool unknown and training only on the confident labels.

The strategy for this pool is **not yet decided**. In particular, an open question is whether the model needs labels for all 631k clips, or whether a smaller high-quality labeled subset (confident meaningful + confident not_meaningful) is sufficient to train a good detector. This section will be completed once the approach is settled.

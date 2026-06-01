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

## The unknown pool: finding confident negatives (in progress)

After the two meaningful carve-outs, 512,377 clips remain unknown — clips where BirdNET named no species and no simulation event occurred. The listening study (see `DIAGNOSIS_LABELS.md`) established this pool is **not** pure background: it contains real birds and insects that BirdNET missed.

The model has plenty of confident *meaningful* labels (118,940) but **zero** confident *not_meaningful* labels. A detector needs both classes. So the open problem is specifically: **how to mine confident background (not_meaningful) clips.** Two signal-based approaches were tested and rejected.

### Rejected signal 1: BirdNET confidence

BirdNET confidence is non-zero only when it names a species. Since all named-species clips were already carved out, every clip in the unknown pool has confidence exactly 0. The signal is a constant across the pool and carries no information for separating background from missed-bird clips. Unusable.

### Rejected signal 2: model score

The trained model outputs a probability (0–1) per clip. Clips it scores very low are its candidates for background. An audit of 80 unknown clips, stratified across four low-score bands (0.00–0.01, 0.01–0.03, 0.03–0.05, 0.05–0.10), 20 per band, was listened to and tagged meaningful / background.

Result: **42 meaningful, 38 background — roughly half-and-half, with no purity gradient.** The lowest band (0.00–0.01, where the model is most confident the clip is background) was actually 65% meaningful (13 of 20). A stricter threshold does not help.

Conclusion: the model **cannot reliably identify background.** It is a one-sided detector — good at recognizing meaningful sound (the false-positive study showed ~93% of its flagged clips were genuinely meaningful), but unreliable at the background end. The likely cause is that it was trained on contaminated negatives (the old BirdNET-<0.3 sample, itself full of missed meaningful sound), so it never learned a real concept of background. This also explains the original over-prediction problem. Model-based negative mining is rejected.

### Chosen approach: harvest background from raw recordings by quiet-stretch detection

Both learned signals (BirdNET, model) failed, so the chosen method is independent of both. Negatives will be harvested from the raw continuous recordings (the multi-hour WAV files the 3-second clips were segmented from):

1. Compute acoustic energy (RMS) over the recording in short windows to locate continuous quiet stretches automatically.
2. Present the quietest candidate windows for listening confirmation (spot-check a few seconds of each).
3. For confirmed-background windows, map the time-window to all 3-second clips inside it (recording start time + offset) and label them not_meaningful — the same time-window-to-clips mechanism used for simulation events.

Properties of this approach:
- **Same-distribution:** negatives come from the same AudioMoth hardware and deployment as the positives, avoiding the source-mismatch trap that arises from importing outside audio.
- **Independent:** relies on signal processing and human listening, not on the failed model/BirdNET signals.
- **Reliable:** confirmed by ear in continuous context, which is more efficient and trustworthy than judging isolated clips.

Known limitation: energy-based detection finds *quiet* background (nighttime silence, gentle ambience) but misses *loud* background such as heavy rain. This is acceptable — the goal is to harvest a clean batch of confident negatives, not to catch all background. Targeting is focused on low-activity recorders (e.g., Audio_Moth_3 and Audio_Moth_4, ~4–5% meaningful) and quiet time windows.

This step is **not yet executed.** This section will be updated with results (windows confirmed, negative clips harvested) once the harvesting is done.

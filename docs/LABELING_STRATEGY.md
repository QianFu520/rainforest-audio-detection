# Labeling Strategy

## Purpose

This document defines how each of the ~631k clips is labeled as **meaningful**, **not_meaningful**, or **unknown** for the audio detection model. It records the decisions that are settled and explicitly marks the part that is still open.

The previous labeling approach (BirdNET confidence >= 0.3 as the meaningful/not-meaningful oracle) was rejected. See `DIAGNOSIS_LABELS.md`: a listening study found that ~93% of the model's apparent false positives were real meaningful sound that BirdNET missed at threshold 0.3. BirdNET at 0.3 is not a trustworthy oracle for "meaningful."

## Definition of "meaningful"

A clip is **meaningful** if it contains bird sound, other animal sound, or human-activity sound (speech, footsteps, vehicles, gunshots, chainsaws). Insects are treated as meaningful: they are animal sound, and they frequently co-occur with bird sound, so including them prevents real bird clips from being filtered out before the downstream classifier.

A clip is **not_meaningful** if it contains only background / ambient sound (rain, wind, silence, or insect-only chorus) with no bird, other animal, or human-activity content.

One important clarification on insects: sustained insect-only chorus with no co-occurring bird or animal call is treated as **not_meaningful** — nothing in such a clip is worth routing to BirdNET. This is distinct from clips where insects co-occur with bird calls; those remain **meaningful**. The model can learn this distinction because the meaningful class contains many clips where bird calls sit on top of an insect background (the birdnet_species clips were captured in real field conditions where insect sound is nearly always present).

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
| not_meaningful | background_energy | 94 |
| not_meaningful | background_flatness | 2,903 |
| not_meaningful | model_inference_v1 | 3,500 |
| unknown | unlabeled | 505,880 |
| **Total** | | **631,317** |

Confident meaningful labels: **118,940**. Confident not_meaningful labels: **6,497**. Remaining unknown: **505,880** (~80%).

## The unknown pool: finding confident negatives (in progress)

After the two meaningful carve-outs, 512,377 clips remain unknown — clips where BirdNET named no species and no simulation event occurred. The listening study (see `DIAGNOSIS_LABELS.md`) established this pool is **not** pure background: it contains real birds and insects that BirdNET missed.

The model has plenty of confident *meaningful* labels (118,940) but **zero** confident *not_meaningful* labels. A detector needs both classes. So the open problem is specifically: **how to mine confident background (not_meaningful) clips.** Two signal-based approaches were tested and rejected.

### Rejected signal 1: BirdNET confidence

BirdNET confidence is non-zero only when it names a species. Since all named-species clips were already carved out, every clip in the unknown pool has confidence exactly 0. The signal is a constant across the pool and carries no information for separating background from missed-bird clips. Unusable.

### Rejected signal 2: model score

The trained model outputs a probability (0–1) per clip. Clips it scores very low are its candidates for background. An audit of 80 unknown clips, stratified across four low-score bands (0.00–0.01, 0.01–0.03, 0.03–0.05, 0.05–0.10), 20 per band, was listened to and tagged meaningful / background.

Result: **42 meaningful, 38 background — roughly half-and-half, with no purity gradient.** The lowest band (0.00–0.01, where the model is most confident the clip is background) was actually 65% meaningful (13 of 20). A stricter threshold does not help.

Conclusion: the model **cannot reliably identify background.** It is a one-sided detector — good at recognizing meaningful sound (the false-positive study showed ~93% of its flagged clips were genuinely meaningful), but unreliable at the background end. The likely cause is that it was trained on contaminated negatives (the old BirdNET-<0.3 sample, itself full of missed meaningful sound), so it never learned a real concept of background. This also explains the original over-prediction problem. Model-based negative mining is rejected.

### Chosen approach: harvest background from raw recordings using acoustic scanning

Both learned signals (BirdNET, model) failed, so the chosen method is independent of both. Negatives are harvested from the raw continuous recordings (the multi-hour WAV files the 3-second clips were segmented from) using two complementary acoustic scans, followed by human spot-check confirmation.

**Properties of this approach:**
- **Same-distribution:** negatives come from the same AudioMoth hardware and deployment as the positives, avoiding source-mismatch from importing outside audio.
- **Independent:** relies on signal processing and human listening, not on the failed model/BirdNET signals.
- **Reliable:** confirmed by ear in continuous context, which is more efficient and trustworthy than judging isolated 3-second clips.

#### Step 1: RMS energy scan (quiet-stretch detection)

Compute RMS (root mean square energy — a measure of loudness) per 3-second window across the full recording by streaming the file in chunks without loading it into memory. Find continuous stretches where every window stays below the recording's own 5th-percentile RMS threshold for 2+ minutes. Report the top 10 quietest candidates with clock times and duration.

**Result on Audio_Moth_3 March 19 evening (20250319_180002.WAV, 12.4 hours):** 3 candidates found, all clustered between 04:50–05:12 AM (1–2 hours before dawn — the only consistently quiet window in this recording). Spot-check by ear:
- Candidate 1 (05:11, 6.6 min): clearly audible bird sound → **rejected**
- Candidate 2 (05:09, 2.1 min): faint/distant bird sound, below BirdNET's detection threshold → **accepted**
- Candidate 3 (04:50, 2.6 min): faint/distant bird sound, below BirdNET's detection threshold → **accepted**

The "faint/distant bird" clips are accepted as not_meaningful because they are already in the unknown pool (BirdNET found nothing in them) and even if the CNN passes them to BirdNET, BirdNET will still find nothing. They are not worth routing to the downstream classifier.

**Clips labeled via RMS scan: 94** (source: `background_energy`)

**Known limitation:** RMS only finds *quiet* background. A loud-but-biologically-empty window (heavy rain, wind) would be rejected because its energy is high. All candidates from this recording were pre-dawn — a temporal bias that would be a problem if the negative class came only from this method.

#### Step 2: Spectral flatness scan

To address the temporal bias and find background candidates at other times of day, a second scan computes **spectral flatness** (Wiener entropy) per 3-second window alongside the energy scan.

Spectral flatness measures the *shape* of the frequency spectrum, not its loudness. It is calculated as the ratio of the geometric mean to the arithmetic mean of the power spectrum, and ranges from 0 to 1:
- **0 (tonal):** energy concentrated in narrow frequency bands — characteristic of bird calls, frog calls, individual insect species
- **1 (flat):** energy spread evenly across all frequencies — characteristic of broadband noise like wind, rain, or the blended wash of a dense multi-species insect chorus

Sustained windows with high flatness (top 30% of the recording's own distribution, held for 2+ minutes) are background candidates regardless of their overall loudness. This allows the method to find loud-but-flat windows that the RMS scan would miss.

**Result on the same recording:** 10 candidates found across a wider time range (02:44–06:06 AM). Spot-check of two long candidates:
- Flatness candidate 3 (04:07, 60.7 min): sustained insect-only chorus, no bird sound detected → **accepted**
- Flatness candidate 6 (03:21, 46.6 min): sustained insect-only chorus, no bird sound detected → **accepted**

These clips contain insect sound but no bird, other animal, or human activity. Per the refined not_meaningful definition above, insect-only clips are not_meaningful for this use case.

**Clips labeled via flatness scan from Audio_Moth_3: 2,094** (source: `background_flatness`)

**Audio_Moth_1, March 17 morning (20250317_093112.WAV, 8.48 hours):** RMS scan found 0 candidates — daytime recordings are too active throughout for any sustained quiet stretch. Flatness scan found 5 candidates, all between 16:41–17:34 (late afternoon). The two longest candidates had very high RMS (0.06–0.07) combined with high flatness — the signature of heavy rain. Spot-check confirmed heavy rain. Both accepted with trimmed starts to skip brief animal calls at the rain onset transition. Rain clips are accepted as not_meaningful because: (a) the meaningful class already contains rain + bird clips from BirdNET detections on rainy days, so the model has positive examples to learn the distinction; (b) the brief calls under heavy rain are below BirdNET's detection threshold — not worth routing.

**Clips labeled via flatness scan from Audio_Moth_1: 809** (source: `background_flatness`)

#### Step 3: Clip mapping

For each confirmed window, the time range is mapped back to 3-second clip names by parsing the timestamp embedded in each clip filename (`Recorder_YYYYMMDD_HHMMSS.wav`) and filtering to clips within the window. The same guard used in the meaningful carve-outs applies: existing confident labels (`human_activity`, `birdnet_species`) are never overwritten. Each recording's mapping cell hardcodes its own `rec_start` so it runs correctly regardless of which raw WAV file is currently loaded in the notebook.

### Model-assisted labeling (iterative expansion)

After training TinyCNN v1 on the acoustic-scan negatives (2,997 clips), the model was run on all 509,380 unknown clips to find additional not_meaningful candidates. This is a self-reinforcing loop: the model trained on confirmed background labels is used to surface more background candidates from the unknown pool, which are then verified by ear and added to the training set.

**Inference:** all 509,380 unknown clips were scored in one pass (~31 minutes on Apple Silicon MPS). Each clip's `not_meaningful_prob` (probability of being background) was saved to `outputs/inference_v1.csv`.

**Candidate selection:** two thresholds were used based on observed precision from a spot-check audit:
- Audio_Moth_3 clips: `not_meaningful_prob >= 0.99` (92% precision in 50-clip audit)
- All other recorders: `not_meaningful_prob >= 0.95` (87% precision in 30-clip audit)

**Spot-check audit (80 clips, 50 from AM3 + 30 from other recorders):**
- 72/80 confirmed background (90% overall precision)
- 8/80 tagged as meaningful (primarily faint bird sound behind insect chorus in AM3 clips)
- Acoustic types found: insect-only chorus (AM3, nighttime), heavy rain (AM2, AM4, AM6), river/flowing water (AM3)
- 10% label noise is acceptable at this stage — the model is robust to some noise in the negative class

**Result:** 3,500 new not_meaningful clips labeled across all 6 recorders (source: `model_inference_v1`). Not_meaningful total increased from 2,997 to **6,497** — more than double, with coverage now spanning all recorders and multiple acoustic background types.

**Guard:** clips already carrying a confident label (`human_activity`, `birdnet_species`, `background_energy`, `background_flatness`) were never overwritten.

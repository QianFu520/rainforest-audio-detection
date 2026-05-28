# Diagnosis: Labeling Problem in the Audio Detection Pipeline

## Summary

A CNN-based audio detection model was developed and evaluated on the full clip-level dataset (~631k clips). On balanced test data the model reached 97% recall and ~80% precision, but on the full dataset precision collapsed to 17% while recall held at 97%. A structured listening study on 270 sampled false positives revealed that ~93% of them were correct detections that the labeling oracle had missed. The model architecture is not the problem; the labeling strategy is the failure point.

## Context

The model's task is binary detection: classify each 3-second clip as **meaningful** (bird sound, other animal sound, or human activity) or **not meaningful** (background, ambient, weather). The dataset is 631,321 non-overlapping 3-second clips drawn from six AudioMoth recorders deployed in a Costa Rican rainforest.

Pseudo-labels were generated from two oracles:
- **BirdNET** confidence ≥ 0.3 for bird sound
- **YAMNet** confidence ≥ 0.3 for human activity classes

The TinyCNN was trained on mel spectrograms with a 1:1 balanced sample (86,585 meaningful + 86,585 non-meaningful), with the non-meaningful sample drawn per-recorder in proportion to each recorder's positive count.

## Initial Result and the Mismatch

| Eval Set | Recall | Precision |
|---|---|---|
| Balanced test (2,400 clips) | 97% | ~80% |
| Full dataset (631k clips) | 97.3% | 16.9% |

Per-recorder precision varied from 5% (Audio_Moth_3, 4% positive rate) to 41% (Audio_Moth_1, 36% positive rate), tracking each recorder's underlying positive prevalence. Recall stayed in the 93–99% range across all recorders.

The initial hypothesis was that the model had been trained on too narrow a sample of non-meaningful clips and was over-predicting at deployment.

## Listening Study

A stratified sample of 270 false positives (15 per recorder × 3 model-confidence bins × 6 recorders) was extracted and inspected by listening to the audio and viewing the mel spectrogram of each clip.

**Categorization results across all bins (270 clips):**

| Bin | Bird | Insects | Bird + Insects (meaningful) | Background (not meaningful) |
|---|---|---|---|---|
| High conf (≥0.8) | 63 (70%) | 12 (13%) | 75 (83%) | 14 (16%) |
| Mid conf (0.5–0.8) | 66 (73%) | 20 (22%) | 86 (96%) | 4 (4%) |
| Low conf (0.3–0.5) | 59 (66%) | 31 (34%) | 90 (100%) | 0 (0%) |
| **Total** | **188 (70%)** | **63 (23%)** | **251 (93%)** | **18 (7%)** |

The audited clips frequently contained layered acoustic scenes: rain, running water, and human activity often co-occurred with bird and insect sound, but biological signal was audibly present in 93% of the sample. The model correctly flagged these as meaningful; BirdNET had returned confidence < 0.3.

Three observations from the listening study:

1. **Layered scenes dominate.** Most clips contain multiple sound sources at once. Detecting biological signal under rain or water requires the model to recognize structure embedded in complex audio, not isolated calls.
2. **Bird content is consistent across confidence bins.** The high, mid, and low bins all show ~65–73% clear bird content. Lower model confidence does not correspond to more model errors — if anything, the high-confidence bin contained slightly more genuine false positives (rain, device noise) than the lower bins.
3. **Insects co-occur with birds.** Many clips initially tagged as insects on first listen revealed bird sound on second listen. Insects are treated as meaningful in this project because (a) they constitute "other animal sound" by the project definition and (b) they often co-occur with bird sound, so including them as positive prevents real bird clips from being filtered out.

## Reinterpreted Metrics

Treating audited hidden true positives as correct detections:

- Reported full-dataset precision: 16.9%
- Estimated true precision (extrapolating from the 93% audited hidden-TP rate): ~94%

The reported precision was almost entirely an artifact of an unreliable label, not of model error.

## Diagnosis

The labeling strategy is the failure point. Specifically:

1. **BirdNET at confidence ≥ 0.3 misses the majority of bird sound in this dataset.** It particularly misses birds in layered scenes — i.e., the realistic deployment condition. In the listening sample, ~70% of false positives contained clear bird sound that BirdNET did not detect at threshold 0.3.

2. **YAMNet contributed almost nothing to the positive class.** It flagged 735 clips total (0.12% of the dataset) — 730 Speech + 5 Chainsaw. The simulation ground truth (described below) shows ≥10,000 clips with actual human activity. YAMNet's recall is far too low to be useful.

3. **A simulation ground truth exists and is more trustworthy.** During field deployment, 63 staged events (Human Presence, Vehicle, Gunshot, Chainsaw) were performed at known times near specific recorders. This produces a gold-standard label for ~10,871 clips of human activity, independent of any oracle.

## Implications

- The model architecture is retained as-is. The TinyCNN learned to detect meaningful sound effectively even under a noisy label.
- YAMNet is dropped from the labeling pipeline.
- The simulation event table replaces YAMNet for human activity ground truth.
- BirdNET at 0.3 is replaced as the bird-side oracle. A new labeling strategy is required (to be designed in `LABELING_STRATEGY.md`).
- Training will be redone on labels reconstructed from these decisions. The model is retrained, not redesigned.

## Open Questions Carried Forward

- The 7% genuine false positives (18 clips out of 270) are concentrated in device-noise and rain. These represent a real model weakness that targeted negative examples may improve.
- The 2,331 false negatives have not been audited. A small targeted listening study (~30–40 clips, weighted toward high-BirdNET-confidence misses) is planned to confirm whether the model has any genuine recall weakness.
- BirdNET-positive clips have not been audited. The positive class's label quality has not been independently verified and may also contain noise.

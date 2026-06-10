# Model Training

## Purpose

This document describes how the TinyCNN binary classifier is trained, the decisions behind each choice, what failed, and what worked. It complements `LABELING_STRATEGY.md`, which covers how the training labels were produced.

The model's job is a binary upstream filter: given a 3-second audio clip, predict **meaningful** (contains bird, animal, or human-activity sound worth routing to BirdNET) or **not_meaningful** (background only, safe to discard).

## Input representation: mel spectrograms

Raw audio waveforms are not fed directly to the CNN. Each 3-second clip is first converted to a **log mel spectrogram** — a 2D image where the x-axis is time, the y-axis is frequency (on the mel scale), and brightness represents energy. This representation is standard in audio classification because:

- It converts the classification problem into image classification, which CNNs handle well
- The mel scale compresses high frequencies and stretches low ones, matching how biological hearing works and making bird call features more visually distinct
- Log scaling (decibels) compresses the dynamic range, making quiet features visible alongside loud ones

**Parameters used:**

| Parameter | Value | Reason |
|---|---|---|
| Sample rate | 48,000 Hz | AudioMoth native rate — no resampling needed |
| Duration | 3.0 s | Matches clip length |
| Mel bins (N_MELS) | 128 | Sufficient frequency resolution for bird calls |
| FFT window (N_FFT) | 1,024 | ~21ms window — resolves bird call structure |
| Hop length | 512 | ~11ms step — adequate temporal resolution |
| Min frequency | 50 Hz | Avoids DC noise |
| Max frequency | 16,000 Hz | Covers all biological sounds of interest |
| Time frames | 256 | Cropped from 282 (removes ~0.3s at end) |

The time axis is cropped to exactly 256 frames to satisfy the MPS (Apple Silicon GPU) backend constraint that `AdaptiveAvgPool2d` input sizes must be divisible by output sizes.

**Pre-computation:** Computing mel spectrograms on the fly during training (loading audio + running librosa per clip, per batch) took ~10 minutes per epoch — too slow. All 121,937 spectrograms were pre-computed once and saved as `.npy` files to `outputs/spectrograms/`. Subsequent training epochs read pre-computed files, reducing epoch time to ~3 minutes.

## Model architecture: TinyCNN

A lightweight CNN designed for binary classification on mel spectrograms. Defined in `src/model/architecture.py`.

**Structure:**

```
Input: (batch, 1, 128, 256)   ← single-channel mel spectrogram
│
├── Conv block 1: Conv2d(1→16, 3×3) → BatchNorm → ReLU → MaxPool2d(2)
├── Conv block 2: Conv2d(16→32, 3×3) → BatchNorm → ReLU → MaxPool2d(2)
├── Conv block 3: Conv2d(32→64, 3×3) → BatchNorm → ReLU → MaxPool2d(2)
│
├── AdaptiveAvgPool2d((8, 16))   ← forces spatial size to 8×16
├── Flatten → Linear(8192, 64) → ReLU → Dropout(0.3)
└── Linear(64, 1)   ← raw logit, sigmoid applied externally
```

**Architecture decisions:**
- **BatchNorm after each conv layer** — stabilises training, reduces sensitivity to learning rate
- **Dropout(0.3)** — regularisation to reduce overfitting
- **AdaptiveAvgPool2d** — added to make the model input-size agnostic; the classifier's linear layer always receives a fixed 8×16 spatial map regardless of spectrogram dimensions
- **Sigmoid removed from output** — `BCEWithLogitsLoss` applies sigmoid internally for numerical stability. Sigmoid is applied explicitly during inference to recover probabilities

## Class imbalance

The training set is severely imbalanced: ~40 meaningful clips for every 1 not_meaningful clip. A model trained naively on this data learns to predict "meaningful" for everything and achieves ~97% accuracy while detecting zero background clips.

**Solution: `BCEWithLogitsLoss` with `pos_weight`**

PyTorch's `BCEWithLogitsLoss` accepts a `pos_weight` parameter that scales the loss contribution of the positive class (meaningful, y=1). Setting `pos_weight < 1` downweights meaningful errors, making not_meaningful errors relatively more costly:

```
pos_weight = n_not_meaningful / n_meaningful = 2,397 / 92,850 ≈ 0.026
```

This makes the total loss contribution from meaningful and not_meaningful clips roughly equal during training, forcing the model to learn both classes.

**Optimizer:** Adam, learning rate 1e-3.

## Train/validation split

The 121,937 labeled clips are split 80/20 into training and validation sets.

**Meaningful clips: split by recorder-date group.** All clips from a given recorder on a given day go entirely into training or entirely into validation. This prevents data leakage — neighbouring 3-second clips (recorded 3 seconds apart) are acoustically nearly identical, so if clip A is in training and clip A+3s is in validation, the model is essentially tested on its training data.

**Not_meaningful clips: random clip-level 80/20 split.** Background clips within a sustained quiet window all sound essentially identical, so clip-level splitting is safe. More importantly, this ensures both acoustic types of not_meaningful clips (nighttime insect-only chorus and afternoon rain) appear in both training and validation.

### Lesson learned: acoustic diversity in splits matters

The first split attempt assigned entire recorder-date groups to each split:
- Training not_meaningful: Audio_Moth_3 March 20 nighttime (insect-only chorus)
- Validation not_meaningful: Audio_Moth_1 March 17 afternoon (heavy rain)

The model was trained exclusively on insect-only background and evaluated on rain background — two acoustically completely different sounds. Result: **0 out of 809 not_meaningful validation clips detected** (0% recall). The model learned "this insect pattern = not_meaningful" but had no concept of rain.

After switching to clip-level splitting for not_meaningful clips (so both rain and insect clips appear in both splits), recall jumped from 0% to 99.5%.

**Rule going forward:** when the not_meaningful class has multiple acoustically distinct subtypes (insect, rain, wind, silence), each subtype must be represented in both training and validation. A split that separates acoustic types is worse than useless — it produces misleading evaluation metrics.

## Results: TinyCNN v1

Trained for 10 epochs on 95,247 clips (92,850 meaningful + 2,397 not_meaningful).

| Metric | Value |
|---|---|
| Val accuracy | 99.92% |
| Not_meaningful precision | 96.9% |
| Not_meaningful recall | 99.5% |
| Not_meaningful F1 | 0.982 |
| Meaningful F1 | 1.000 |

**Confusion matrix (validation set, 26,690 clips):**

|  | Predicted not_meaningful | Predicted meaningful |
|---|---|---|
| Actual not_meaningful | 597 | 3 |
| Actual meaningful | 19 | 26,071 |

Val loss was stable throughout training (0.0003–0.0009), with no sign of the overfitting seen in the first attempt. Model weights saved to `outputs/models/tinycnn_v1.pth`.

**Caveat:** validation not_meaningful clips come from the same recordings as training not_meaningful clips (different 3-second windows, same acoustic conditions). A truly out-of-sample test requires running the model on recordings never seen during labeling. This is the next step.

## Results: TinyCNN v2

After v1, the model was run on all 509,380 unknown clips (notebook `06_inference_labeling.ipynb`). High-confidence not_meaningful predictions (AM3 at prob ≥ 0.99, all others at prob ≥ 0.95) were spot-checked by ear — 80 clips sampled, 72/80 confirmed background (90% precision). 3,500 new not_meaningful clips were labeled across all 6 recorders, covering three acoustic types: insect-only chorus, heavy rain, and river/flowing water. Total not_meaningful grew from 2,997 to 6,497.

V2 was trained on 98,047 clips (92,850 meaningful + 5,197 not_meaningful). Class ratio: 18:1 (down from 40:1).

| Metric | v1 | v2 |
|---|---|---|
| Val accuracy | 99.92% | 99.90% |
| Not_meaningful precision | 96.9% | **98.3%** |
| Not_meaningful recall | 99.5% | **99.6%** |
| Not_meaningful F1 | 0.982 | **0.989** |
| Val not_meaningful support | 600 clips | **1,300 clips** |

**Confusion matrix (validation set, 27,390 clips):**

|  | Predicted not_meaningful | Predicted meaningful |
|---|---|---|
| Actual not_meaningful | 1,295 | 5 |
| Actual meaningful | 23 | 26,067 |

Every metric improved over v1. The val set now has 1,300 not_meaningful clips from all 6 recorders — a much more robust evaluation. Val loss stable throughout (0.0002–0.0022), no overfitting. Model weights saved to `outputs/models/tinycnn_v2.pth`.

## Evaluation strategy: per-recorder audit before production labeling

Val metrics alone are not sufficient to trust a model for production inference. All v2 val clips came from the same recordings that were used for labeling — the model has "seen" the acoustic conditions at those locations. Before using v2 to label the entire 505,880-clip unknown pool, we need to verify that its not_meaningful predictions are correct across all 6 recorders.

**Why a val set is not enough:** every recorder/date combo in the dataset has at least some labeled clips (the minimum is ~500). So there is no truly "held-out" recorder. However, recorders with fewer labeled clips (e.g., Audio_Moth_4 March 17: 517 labeled out of 15,383 total) are the closest proxy — the model has seen very little from those locations.

**Evaluation approach:**
1. Run v2 on all 505,880 unknown clips → save `inference_v2.csv`
2. Sample **10 clips per recorder** (60 total) from high-confidence not_meaningful predictions (prob ≥ 0.95) — this ensures all 6 locations are represented equally
3. Listen to all 60 clips by ear and tag each as `background`, `meaningful`, or `unsure`
4. Check per-recorder precision — if any recorder is below ~85%, it signals that v2 is making errors at that location and targeted negatives should be collected there before production labeling

**Why equal sampling per recorder:** the v1 audit sampled 50 from AM3 and only 30 from all other recorders combined. This was biased toward AM3 and gave weak coverage of AM4, AM5, AM6. Equal per-recorder sampling catches location-specific failure modes.

**Decision rule:**
- All recorders ≥ 85% precision → proceed with production labeling using v2
- Any recorder < 85% → investigate that recorder's predictions, collect targeted negatives, retrain v3

## v2 audit results

Ran v2 on all 505,880 unknown clips. Sampled 10 clips per recorder (60 total) from high-confidence not_meaningful predictions (prob ≥ 0.95) and tagged each by ear.

| Recorder | Background | Meaningful | Precision |
|---|---|---|---|
| Audio_Moth_1 | 7/10 | 3 | 70% ← below threshold |
| Audio_Moth_2 | 8/10 | 2 | 80% ← below threshold |
| Audio_Moth_3 | 10/10 | 0 | 100% |
| Audio_Moth_4 | 10/10 | 0 | 100% |
| Audio_Moth_5 | 9/10 | 1 | 90% |
| Audio_Moth_6 | 10/10 | 0 | 100% |
| **Overall** | **54/60** | **6** | **90%** |

AM3, AM4, AM5, AM6 all passed (≥ 85%). AM1 and AM2 failed.

**What the false positives sounded like:** brief bird calls behind loud background noise (rain or strong insect chorus) — not very faint, not very clear, short duration. The model heard the dominant background and predicted not_meaningful, but the bird call was still audible underneath. Whether BirdNET would detect these calls through the heavy background is uncertain, making these genuine edge cases.

**Initial decision:** raise threshold to 0.99 for AM1/AM2 and re-audit before labeling.

### AM1/AM2 re-audit at prob ≥ 0.99

Sampled 10 clips each from AM1 and AM2 at the stricter 0.99 threshold and tagged by ear.

| Recorder | Background | Meaningful | Precision |
|---|---|---|---|
| Audio_Moth_1 | 9/10 | 1 | 90% ✓ passed |
| Audio_Moth_2 | 9/10 | 1 | 90% ✓ passed |

Both passed. The 2 remaining false positives were: one clip with a human voice, and one with a brief bird call — genuine edge cases at the boundary of detection.

**Final thresholds:**
- Audio_Moth_1, Audio_Moth_2: prob ≥ 0.99
- Audio_Moth_3, Audio_Moth_4, Audio_Moth_5, Audio_Moth_6: prob ≥ 0.95

**Final label counts after v2 inference labeling:**

| Source | Clips |
|---|---|
| model_inference_v2 | 4,716 |
| model_inference_v1 | 3,500 |
| background_flatness | 2,903 |
| background_energy | 94 |
| **Total not_meaningful** | **11,213** |

Labels remaining as unknown: 501,164.

## Results: TinyCNN v3

Trained on 130,153 clips (118,940 meaningful + 11,213 not_meaningful). Class ratio: ~10:1 (down from 18:1 in v2). `pos_weight` adjusted automatically to 0.097.

| Metric | v1 | v2 | v3 |
|---|---|---|---|
| Val accuracy | 99.92% | 99.90% | 99.80% |
| Not_meaningful precision | 96.9% | 98.3% | **98.0%** |
| Not_meaningful recall | 99.5% | 99.6% | **100%** |
| Not_meaningful F1 | 0.982 | 0.989 | **0.990** |
| Val not_meaningful support | 600 | 1,300 | **2,243** |

**Confusion matrix (validation set, 28,333 clips):**

|  | Predicted not_meaningful | Predicted meaningful |
|---|---|---|
| Actual not_meaningful | 2,243 | 0 |
| Actual meaningful | 46 | 26,044 |

**Key improvement over v2:** recall reached 100% — zero not_meaningful clips missed. The trade-off is 46 meaningful clips incorrectly filtered (vs 23 in v2), but that is only 0.18% of meaningful clips. The val not_meaningful support nearly doubled (2,243 vs 1,300), making this the most trustworthy evaluation so far.

Val loss was slightly noisier than v2 (0.0006–0.0027 across epochs vs a smoother curve in v2), likely due to the more acoustically diverse training data spanning all 6 recorders. No sign of overfitting. Model weights saved to `outputs/models/tinycnn_v3.pth`.

## v3 audit results

Ran v3 on all 501,164 unknown clips. Sampled 10 clips per recorder (60 total) from high-confidence not_meaningful predictions (prob ≥ 0.95) and tagged each by ear.

| Recorder | Background | Meaningful | Precision |
|---|---|---|---|
| Audio_Moth_1 | 9/10 | 1 | 90% ✓ passed |
| Audio_Moth_2 | 9/10 | 1 | 90% ✓ passed |
| Audio_Moth_3 | 10/10 | 0 | 100% |
| Audio_Moth_4 | 10/10 | 0 | 100% |
| Audio_Moth_5 | 10/10 | 0 | 100% |
| Audio_Moth_6 | 9/10 | 1 | 90% ✓ passed |
| **Overall** | **57/60** | **3** | **95%** |

All 6 recorders passed the 85% gate. Notably, AM1 improved from 70% → 90% and AM2 from 80% → 90% compared to the v2 audit at the same threshold — v3 is significantly better calibrated for those locations.

**Final threshold: prob ≥ 0.95 for all recorders** (no per-recorder split needed).

**Result:** 5,632 new not_meaningful clips labeled (source: `model_inference_v3`). Not_meaningful total increased from 11,213 to **16,845**. Unknown pool: 495,532.

## Results: TinyCNN v4

Trained on 135,785 clips (118,940 meaningful + 16,845 not_meaningful). Class ratio: ~7:1. `pos_weight`: 0.145.

| Metric | v1 | v2 | v3 | v4 |
|---|---|---|---|---|
| Val accuracy | 99.92% | 99.90% | 99.80% | 99.61% |
| Not_meaningful precision | 96.9% | 98.3% | **98.0%** | 96.7% |
| Not_meaningful recall | 99.5% | 99.6% | **100%** | **100%** |
| Not_meaningful F1 | 0.982 | 0.989 | **0.990** | 0.983 |
| Val not_meaningful support | 600 | 1,300 | 2,243 | 3,369 |

**Confusion matrix (validation set, 29,459 clips):**

|  | Predicted not_meaningful | Predicted meaningful |
|---|---|---|
| Actual not_meaningful | 3,369 | 0 |
| Actual meaningful | 116 | 25,974 |

**V4 shows diminishing returns and slight regression.** Recall held at 100%, but precision dropped from 98.0% to 96.7% and F1 regressed from 0.990 to 0.983. The number of meaningful clips incorrectly filtered rose from 46 to 116. This is a known pattern in iterative self-labeling: each round introduces ~5% label noise, and after enough iterations that noise begins to accumulate and hurt precision.

**Decision: v3 is the final production model.** V4 confirms the iterative loop has converged — adding more self-labeled data no longer improves the model. Model weights saved to `outputs/models/tinycnn_v4.pth` for reference.

## Final model: TinyCNN v3

- **Weights:** `outputs/models/tinycnn_v3.pth`
- **Not_meaningful precision:** 98.0% — 2% of clips the model filters out are actually meaningful
- **Not_meaningful recall:** 100% — no real background clip is missed
- **F1:** 0.990
- **Training data:** 11,213 not_meaningful + 118,940 meaningful (10:1 ratio)

## Next steps

1. **Production scripts** — write clean Python scripts for the full inference pipeline using v3
2. **Final inference run** — use v3 to label the remaining 495,532 unknown clips for downstream BirdNET processing

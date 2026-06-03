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

## Next steps

1. **Evaluate on truly unseen recordings** — test on a recorder/date combination not present in any labeled data to verify generalization across different Costa Rican rainforest locations
2. **Further iteration if needed** — if evaluation on unseen data reveals failures, collect targeted negatives and retrain v3
3. **Production scripts** — once the model is validated, write clean Python scripts for the full inference pipeline

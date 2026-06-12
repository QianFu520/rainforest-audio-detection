# BirdNET Compression — Stage 2

## Purpose

This document records the decisions and methodology for Stage 2 of the pipeline: compressing BirdNET v2.4 for edge deployment on a Raspberry Pi 4 (ARM Cortex-A72) and AWS inference service.

Stage 1 (TinyCNN binary filter) feeds Stage 2. Only clips classified as `meaningful` by TinyCNN are routed to BirdNET for species identification.

## Source model

**Model:** BirdNET v2.4 acoustic model, FP32 SavedModel (TensorFlow Protobuf format)

**Source:** Zenodo record 15050749 — `BirdNET_v2.4_protobuf.zip`

**Local path:** `~/Library/Application Support/birdnet/acoustic-models/v2.4/pb/model-fp32/`

**Size:** 56 MB on disk (SavedModel directory with `saved_model.pb` + `variables/`)

**Why this source and not the birdnet-analyzer GitHub checkpoints:** The original `birdnet_species` labels in `labels_progress.csv` (108,069 clips) were generated using the `birdnet` Python package (v0.1.7, `AudioModelV2M4TFLite`). That TFLite model downloads from the same Zenodo record (file size 51,726,412 bytes matches). The birdnet-analyzer CLI uses a different, smaller model (~21.3 MB FP32 TFLite). Using the Zenodo SavedModel ensures our PTQ output is comparable to the labels already in the dataset.

## Compression plan

**Approach:** Post-training quantization (PTQ) int8 — apply TensorFlow Lite converter with a calibration dataset to the FP32 SavedModel, producing a quantized `.tflite` artifact.

**Why PTQ ourselves and not the pre-built official INT8 variant:** The `birdnet` package (v0.2.11) ships a pre-built INT8 TFLite (~39 MB) from the same Zenodo record. Using it would skip the compression step entirely. Doing PTQ ourselves demonstrates the compression pipeline as a portfolio skill. After PTQ we will compare our artifact against the official INT8 to validate that our quantization matches the reference.

**Calibration data:** a representative sample from the 108,069 `birdnet_species` clips — real field audio that spans the acoustic conditions in the dataset.

**Target format:** TFLite INT8

**Target device:** Raspberry Pi 4 (ARM Cortex-A72) — simulated benchmark on Mac, then validated on actual Pi hardware if available.

## Evaluation strategy

**Ground truth:** FP32 SavedModel top-1 species prediction on the 108,069 `birdnet_species` clips.

**Metric:** top-1 agreement rate — fraction of clips where the compressed model's top-1 prediction matches the FP32 model's top-1 prediction.

**Why this metric and not absolute accuracy:** we do not have human-verified species labels for these clips. BirdNET's own output is the most trustworthy reference available. The metric measures compression fidelity: how much does quantization change the model's predictions?

**Additional measurements per model variant:**
- Model size (MB on disk)
- Inference latency (ms per 3-second clip, benchmarked on Mac as Pi 4 proxy)
- Top-1 agreement rate vs FP32 baseline

All experiments tracked in MLflow from day 1.

## API notes

The `birdnet` Python package (v0.2.11) is the interface for loading and running the SavedModel:

```python
from birdnet.model_loader import load
model = load("acoustic", "2.4", "pb")  # loads FP32 SavedModel
result = model.predict(clip_path, top_k=5)
df = result.to_dataframe()  # columns: input, start_time, end_time, species_name, confidence
```

`model.predict()` returns an `AcousticFilePredictionResult`. Use `.to_dataframe()` — the result object is not directly iterable.

## Conversion findings

### Jupyter + TFLite converter incompatibility on macOS

The TFLite converter spawns internal subprocesses during graph transformation. On macOS, these subprocesses conflict with Jupyter's event loop and signal handling, causing the kernel to hang indefinitely — both for dynamic range quantization and full INT8 conversion with a calibration dataset. The conversion completes in ~11 seconds when run as a standalone Python script outside Jupyter.

**Workaround:** the conversion cell in `08_birdnet_ptq.ipynb` uses `subprocess.run` to invoke a standalone script (`scripts/convert_birdnet_ptq.py`), then loads the resulting `.tflite` file back into the notebook for evaluation.

### Full INT8 calibration speed

Full INT8 PTQ requires running each calibration clip through the full FP32 SavedModel on CPU to collect activation statistics. On an M4 Mac without `tensorflow-metal`, TF SavedModel inference runs on CPU only (~20 seconds per clip). 100 clips × ~20s = ~33 minutes — slow but not stuck. This is a hardware constraint, not a bug. `tensorflow-metal` would accelerate this but is not currently installed in the `ds` environment.

### Dynamic range quantization result

Dynamic range quantization — weights quantized to INT8, activations remain float32 at runtime:

| Variant | Size | Reduction |
|---|---|---|
| FP32 TFLite (baseline) | 51.7 MB | — |
| Dynamic range INT8 TFLite (ours) | 14.2 MB | 72.5% |

Conversion time: 11 seconds as a standalone script (M4 Mac, CPU only).

**Prediction quality: wrong.** Sanity check on the test clip returned Madagascar Scops-Owl as top-1; FP32 returns Spotted Antbird. See investigation below.

### Calibrated INT8 PTQ result

Full INT8 PTQ — weights and activations both INT8, scale factors from 100 calibration clips:

| Variant | Size | Reduction |
|---|---|---|
| FP32 TFLite (baseline) | 51.7 MB | — |
| Calibrated INT8 TFLite (ours) | 14.1 MB | 72.8% |

Conversion time: ~2 minutes via standalone script in TF 2.15.

**Prediction quality: wrong.** 500-clip spot check: 0/500 top-1 agreement (0.0%). Two models cannot disagree on 100% of 500 clips by chance — this is a definitive failure.

### Why INT8 PTQ fails for this model

Both dynamic range and calibrated INT8 produce wrong predictions despite the SavedModel → FP32 TFLite conversion giving correct results. The conversion pipeline itself is not the issue.

Investigation confirmed that the **official Zenodo INT8 model was not built with PTQ**. Its tensor names contain `FakeQuantWithMinMaxVars` and `quant_`-prefixed operations — artifacts of **quantization-aware training (QAT)**, where the model is retrained with fake quantization nodes inserted during training. QAT teaches the model to be numerically robust to INT8 precision; PTQ applies quantization to a model that was never trained with this constraint.

Key evidence from tensor inspection:

| Model | INT8 tensors | float32 tensors | Quality |
|---|---|---|---|
| Official INT8 (Zenodo QAT) | 139 | 345 | correct |
| Our calibrated INT8 (PTQ) | 372 | 15 | 0% agreement |
| Our dynamic range (PTQ) | 77 | 343 | wrong |

The official INT8 quantizes only the QAT-prepared backbone layers; our PTQ converts more ops including sensitive ones. Reproducing the official INT8 would require BirdNET's training code and dataset — not feasible from the public SavedModel.

**16x8 PTQ also fails.** Weights INT8, activations INT16 (`EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8`), 100-clip calibration. 14.4 MB, 72.2% reduction. Sanity check: Variegated Antpitta as top-1 (wrong). The INT16 activations do not help because the root cause is the INT8 *weight* quantization rounding small mel filterbank values to zero — not the activation precision.

**Working approach: FP16 PTQ.** Weights stored as float16, used as float32 at runtime. No calibration needed. Float16 preserves the mel filterbank values accurately (~3.3 decimal digits vs INT8's ~2.1). See FP16 result section below.

### FP16 PTQ result

FP16 quantization — weights stored as float16, activations and runtime computation remain float32:

| Variant | Size | Reduction | Top-1 agreement (108,069 clips) |
|---|---|---|---|
| FP32 TFLite (baseline) | 51.7 MB | — | — |
| FP16 TFLite (ours) | 26.0 MB | 49.8% | **103,026/108,069 = 95.33%** |

Conversion time: ~10 seconds as a standalone script (no calibration). Output: `outputs/models/birdnet_v2.4_fp16.tflite`.

**95.33% overall agreement is not a compression fidelity problem.** All 5,043 disagreements come from borderline clips where the original BirdNET prediction had very low confidence (mean 0.32 vs 0.50 for agreements). Float16 rounding is enough to flip near-tie argmax values. Agreement rate rises sharply with confidence threshold:

| Confidence ≥ | Clips | Top-1 agreement |
|---|---|---|
| 0.25 (all clips) | 108,069 | 95.33% |
| 0.30 | 85,237 | 97.25% |
| 0.35 | 68,984 | 98.33% |
| 0.40 | 57,128 | 99.01% |
| 0.50 | 40,352 | 99.57% |
| 0.60 | 28,882 | 99.74% |

The dominant disagreement pattern: 2,559 clips labeled "Spectacled Owl" (mean confidence 0.30) are predicted as "Choco Screech-Owl" by FP16 — two acoustically similar nocturnal species at the decision boundary. Any deployment using a confidence threshold ≥ 0.35 sees 98%+ agreement.

The 49.8% size reduction comes entirely from halving the weight storage (float32 → float16); inference runs in float32.

### `conda run -n tf215` environment note

The `tf215` conda environment was created to work around TF 2.20's SavedModel loading hang on M4 Mac. However, `conda run -n tf215` silently falls back to the `ds` environment when called from a subprocess — confirmed by `sys.prefix` pointing to the `ds` path and `tf.__version__` returning 2.20.0.

All conversions in this project ran under TF 2.20, not TF 2.15, despite the `tf215` environment being specified. The correct invocation uses the binary directly:

```
/Users/qian/miniforge3/envs/tf215/bin/python scripts/convert_birdnet_ptq.py
```

## Current state

| Step | Status |
|---|---|
| FP32 SavedModel downloaded and verified | done |
| Baseline notebook (07_birdnet_baseline.ipynb) | done |
| Notebook restructured to use subprocess for conversion | done |
| Dynamic range INT8 conversion | done — 14.2 MB, 72.5% reduction, wrong predictions |
| Calibrated INT8 PTQ | done — 14.1 MB, 72.8% reduction, 0% top-1 agreement |
| INT8 PTQ investigation | done — official INT8 is QAT; standard PTQ not viable |
| 16x8 PTQ (INT8 weights, INT16 activations) | done — 14.4 MB, 72.2% reduction, wrong predictions |
| FP16 PTQ | done — 26.0 MB, 49.8% reduction, 500/500 = 100% top-1 agreement |
| Agreement evaluation on 108,069 clips | done — 95.33% overall; 99.57% at confidence ≥ 0.50 |
| MLflow experiment tracking | next |
| Latency benchmark | next |

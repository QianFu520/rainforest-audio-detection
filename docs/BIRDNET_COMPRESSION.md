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

## Current state

| Step | Status |
|---|---|
| FP32 SavedModel downloaded and verified | done |
| Baseline notebook (07_birdnet_baseline.ipynb) | done |
| PTQ int8 conversion | next |
| Calibration dataset selection | next |
| Agreement evaluation on 108,069 clips | next |
| MLflow experiment tracking | next |
| Latency benchmark | next |

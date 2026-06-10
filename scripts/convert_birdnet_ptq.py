"""
Convert BirdNET v2.4 FP32 SavedModel to INT8 TFLite via full INT8 PTQ with calibration.

Must be run as a standalone script — the TFLite converter spawns subprocesses that
conflict with Jupyter's event loop on macOS and cause the kernel to hang.

Usage:
    /Users/qian/miniforge3/envs/tf215/bin/python scripts/convert_birdnet_ptq.py [output_path]

Note: `conda run -n tf215` silently falls back to the ds environment (TF 2.20).
Use the binary path directly to ensure TF 2.15 is used.

Runtime: ~2 minutes on M4 Mac (TF 2.15, CPU calibration, 100 clips)
"""
import os
import sys
import numpy as np
import pandas as pd
import soundfile as sf
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

MODEL_PATH = "/Users/qian/Library/Application Support/birdnet/acoustic-models/v2.4/pb/model-fp32"
FP32_TFLITE_PATH = "/Users/qian/Library/Application Support/birdnet/acoustic-models/v2.4/tf/model-fp32.tflite"
OUTPUT_PATH = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    config.OUTPUTS_DIR, "models", "birdnet_v2.4_int8_calibrated.tflite"
)

N_SAMPLES = 144000  # 3s x 48kHz
N_CALIB = 100
RANDOM_SEED = 42

# Sample calibration clips from birdnet_species, stratified by recorder
labels_df = pd.read_csv(config.LABELS_PROGRESS_PATH)
species_clips = labels_df[labels_df["meaningful_source"] == "birdnet_species"].copy()
n_recorders = species_clips["Recorder"].nunique()
per_recorder = N_CALIB // n_recorders

calib_clips = (
    species_clips
    .groupby("Recorder", group_keys=False)
    .apply(lambda g: g.sample(min(len(g), per_recorder), random_state=RANDOM_SEED))
    .sample(frac=1, random_state=RANDOM_SEED)
    .head(N_CALIB)
    .reset_index(drop=True)
)
print(f"Calibration clips: {len(calib_clips)} across {n_recorders} recorders")


def load_audio(path):
    audio, _ = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if len(audio) < N_SAMPLES:
        audio = np.pad(audio, (0, N_SAMPLES - len(audio)))
    else:
        audio = audio[:N_SAMPLES]
    return audio.reshape(1, N_SAMPLES).astype(np.float32)


def representative_dataset():
    for i, (_, row) in enumerate(calib_clips.iterrows()):
        if i % 10 == 0:
            print(f"  Calibrating clip {i + 1}/{len(calib_clips)}...")
        yield {"inputs": load_audio(row["audio_path"])}


# Full INT8 PTQ — mixed mode: INT8 where possible, float fallback for SE/sigmoid ops
# signature_keys=['basic'] needed because the SavedModel has 2 signatures (basic + embeddings)
converter = tf.lite.TFLiteConverter.from_saved_model(MODEL_PATH, signature_keys=["basic"])
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
    tf.lite.OpsSet.TFLITE_BUILTINS,
]
converter.inference_input_type = tf.float32
converter.inference_output_type = tf.float32

print("Converting... (~30 min on M4 Mac CPU)")
tflite_model = converter.convert()

os.makedirs(os.path.dirname(os.path.abspath(OUTPUT_PATH)), exist_ok=True)
with open(OUTPUT_PATH, "wb") as f:
    f.write(tflite_model)

fp32_mb = os.path.getsize(FP32_TFLITE_PATH) / 1e6
int8_mb = os.path.getsize(OUTPUT_PATH) / 1e6
print(f"FP32 TFLite:    {fp32_mb:.1f} MB")
print(f"INT8 TFLite:    {int8_mb:.1f} MB  (calibrated)")
print(f"Size reduction: {(1 - int8_mb / fp32_mb) * 100:.1f}%")
print(f"Saved: {OUTPUT_PATH}")

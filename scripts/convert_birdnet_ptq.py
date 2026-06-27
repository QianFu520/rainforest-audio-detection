"""
Convert BirdNET v2.4 FP32 SavedModel to FP16 TFLite via PTQ.

FP16 mode: weights stored as float16, used as float32 at runtime. No calibration
needed. Preserves mel filterbank and STFT weights accurately (float16 has ~3.3
decimal digits of precision vs INT8's ~2.1).

INT8-weight approaches (dynamic range, full INT8, 16x8) all fail for this model —
the mel filterbank has small values that INT8 rounds to zero, corrupting the
spectrogram. The official Zenodo INT8 used QAT to work around this.
"""
import os
import sys
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

MODEL_PATH = "/Users/qian/Library/Application Support/birdnet/acoustic-models/v2.4/pb/model-fp32"
FP32_TFLITE_PATH = "/Users/qian/Library/Application Support/birdnet/acoustic-models/v2.4/tf/model-fp32.tflite"
OUTPUT_PATH = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    config.OUTPUTS_DIR, "models", "birdnet_v2.4_fp16.tflite"
)

# FP16 PTQ — weights stored as float16, no calibration needed
# signature_keys=['basic'] needed because the SavedModel has 2 signatures (basic + embeddings)
converter = tf.lite.TFLiteConverter.from_saved_model(MODEL_PATH, signature_keys=["basic"])
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]

print("Converting...")
tflite_model = converter.convert()

os.makedirs(os.path.dirname(os.path.abspath(OUTPUT_PATH)), exist_ok=True)
with open(OUTPUT_PATH, "wb") as f:
    f.write(tflite_model)

fp32_mb = os.path.getsize(FP32_TFLITE_PATH) / 1e6
fp16_mb = os.path.getsize(OUTPUT_PATH) / 1e6
print(f"FP32 TFLite:    {fp32_mb:.1f} MB")
print(f"FP16 TFLite:    {fp16_mb:.1f} MB  (float16 weights)")
print(f"Size reduction: {(1 - fp16_mb / fp32_mb) * 100:.1f}%")
print(f"Saved: {OUTPUT_PATH}")

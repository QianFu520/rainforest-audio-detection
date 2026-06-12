"""
Evaluate top-1 agreement between the FP16 TFLite model and the stored FP32
BirdNET predictions in labels_progress.csv on all birdnet_species clips.

FP32 predictions are already stored in the `species` column from when the clips
were originally labeled — no need to re-run FP32. We only run FP16 TFLite.


Output CSV columns: clip_name, fp32_species, fp16_species, agree
"""
import ast
import os
import sys
import time

import numpy as np
import pandas as pd
import soundfile as sf
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

LABELS_PATH = "/Users/qian/Library/Application Support/birdnet/acoustic-models/v2.4/pb/labels/en_us.txt"
FP16_PATH = os.path.join(config.OUTPUTS_DIR, "models", "birdnet_v2.4_fp16.tflite")
OUTPUT_CSV = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    config.OUTPUTS_DIR, "eval_fp16_agreement.csv"
)

N_SAMPLES = 144000  # 3s × 48kHz — matches SavedModel input shape

with open(LABELS_PATH) as f:
    labels = [line.strip() for line in f]

df = pd.read_csv(config.LABELS_PROGRESS_PATH)
clips = df[df["meaningful_source"] == "birdnet_species"].reset_index(drop=True)
total = len(clips)
print(f"Clips to evaluate: {total}")
print(f"FP16 model:        {FP16_PATH}")
print(f"Output CSV:        {OUTPUT_CSV}")
print()

interp = tf.lite.Interpreter(model_path=FP16_PATH)
interp.allocate_tensors()
inp_d = interp.get_input_details()[0]
out_d = interp.get_output_details()[0]


def load_audio(path):
    audio, _ = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if len(audio) < N_SAMPLES:
        audio = np.pad(audio, (0, N_SAMPLES - len(audio)))
    else:
        audio = audio[:N_SAMPLES]
    return audio.reshape(1, N_SAMPLES).astype(np.float32)


results = []
agree = 0
errors = 0
t0 = time.time()

for idx, (_, row) in enumerate(clips.iterrows()):
    try:
        fp32_species = ast.literal_eval(row["species"])[0]
    except Exception:
        fp32_species = ""

    try:
        audio = load_audio(row["audio_path"])
        interp.set_tensor(inp_d["index"], audio)
        interp.invoke()
        fp16_idx = int(np.argmax(interp.get_tensor(out_d["index"])[0]))
        fp16_species = labels[fp16_idx]
        match = fp32_species == fp16_species
    except Exception as e:
        fp16_species = f"ERROR: {e}"
        match = False
        errors += 1

    if match:
        agree += 1

    results.append({
        "clip_name": row["clip_name"],
        "fp32_species": fp32_species,
        "fp16_species": fp16_species,
        "agree": match,
    })

    if (idx + 1) % 1000 == 0:
        elapsed = time.time() - t0
        rate = (idx + 1) / elapsed
        eta = (total - idx - 1) / rate
        print(
            f"  {idx+1:>6}/{total}  agreement: {agree/(idx+1)*100:.1f}%"
            f"  elapsed: {elapsed/60:.1f}m  ETA: {eta/60:.1f}m"
        )
        os.makedirs(os.path.dirname(os.path.abspath(OUTPUT_CSV)), exist_ok=True)
        pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)

os.makedirs(os.path.dirname(os.path.abspath(OUTPUT_CSV)), exist_ok=True)
pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)

elapsed = time.time() - t0
print(f"\nTop-1 agreement: {agree}/{total} = {agree/total*100:.2f}%")
if errors:
    print(f"Errors (skipped): {errors}")
print(f"Elapsed: {elapsed/60:.1f} minutes")
print(f"Results saved: {OUTPUT_CSV}")

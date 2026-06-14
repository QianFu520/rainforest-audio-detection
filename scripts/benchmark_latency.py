"""
Benchmark inference latency for FP32, FP16, and INT8 BirdNET TFLite models.

Uses synthetic random input to isolate pure inference time from file I/O.
Appends latency_mean_ms and latency_std_ms to existing MLflow birdnet_compression runs.

Usage:
    python scripts/benchmark_latency.py
"""
import os
import sys
import time
import platform

import numpy as np
import tensorflow as tf
import mlflow
from mlflow.tracking import MlflowClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

DB_PATH = os.path.join(config.OUTPUTS_DIR, "mlflow.db")
mlflow.set_tracking_uri(f"sqlite:///{os.path.abspath(DB_PATH)}")

BIRDNET_DIR = os.path.expanduser(
    "~/Library/Application Support/birdnet/acoustic-models/v2.4/tf"
)
MODELS_DIR = os.path.join(config.OUTPUTS_DIR, "models")

MODELS = [
    {"run_name": "fp32_baseline",     "path": os.path.join(BIRDNET_DIR, "model-fp32.tflite")},
    {"run_name": "fp16_ptq",          "path": os.path.join(MODELS_DIR, "birdnet_v2.4_fp16.tflite")},
    {"run_name": "int8_qat_official", "path": os.path.join(BIRDNET_DIR, "model-int8.tflite")},
]

N_WARMUP = 10
N_TIMED = 1000
HOST_DEVICE = platform.processor() or platform.machine()
TFLITE_VERSION = tf.__version__


def make_input(inp):
    rng = np.random.RandomState(42)
    dtype = inp["dtype"]
    shape = inp["shape"]
    if dtype == np.float32:
        return rng.randn(*shape).astype(np.float32)
    if dtype == np.int8:
        return rng.randint(-128, 127, size=shape).astype(np.int8)
    return np.zeros(shape, dtype=dtype)


def run_benchmark(model_path):
    interp = tf.lite.Interpreter(model_path=model_path)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    x = make_input(inp)

    for _ in range(N_WARMUP):
        interp.set_tensor(inp["index"], x)
        interp.invoke()

    times_ms = []
    for _ in range(N_TIMED):
        interp.set_tensor(inp["index"], x)
        t0 = time.perf_counter()
        interp.invoke()
        times_ms.append((time.perf_counter() - t0) * 1000)

    return np.array(times_ms)


client = MlflowClient()
exp = client.get_experiment_by_name("birdnet_compression")
runs = client.search_runs(exp.experiment_id)
run_map = {r.info.run_name: r.info.run_id for r in runs}

print(f"Host:           {HOST_DEVICE}")
print(f"TFLite version: {TFLITE_VERSION}")
print(f"Warm-up:        {N_WARMUP}  Timed: {N_TIMED}")
print()

for meta in MODELS:
    name = meta["run_name"]
    print(f"Benchmarking {name} ...")

    times_ms = run_benchmark(meta["path"])
    mean_ms   = float(np.mean(times_ms))
    std_ms    = float(np.std(times_ms))
    median_ms = float(np.median(times_ms))
    p95_ms    = float(np.percentile(times_ms, 95))

    print(f"  mean={mean_ms:.1f}ms  std={std_ms:.1f}ms  "
          f"median={median_ms:.1f}ms  p95={p95_ms:.1f}ms")

    run_id = run_map[name]
    client.log_metric(run_id, "latency_mean_ms", mean_ms)
    client.log_metric(run_id, "latency_std_ms", std_ms)
    client.log_param(run_id, "n_warmup", N_WARMUP)
    client.log_param(run_id, "n_timed", N_TIMED)
    client.log_param(run_id, "host_device", HOST_DEVICE)
    client.log_param(run_id, "tflite_runtime_version", TFLITE_VERSION)

print("\nDone — latency logged to MLflow.")

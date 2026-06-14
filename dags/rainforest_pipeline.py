"""
Rainforest audio detection — daily batch pipeline DAG.

Stage 1: TinyCNN binary filter (PyTorch) — filters out non-meaningful clips.
Stage 2: BirdNET FP16 TFLite — species classification on meaningful clips only.

Trigger: S3KeySensor watches s3://{S3_BUCKET}/{S3_INPUT_PREFIX}/ for new .wav files.
Storage: S3 input → local temp scratch → DynamoDB predictions + S3 results.
Cleanup: temp dir removed after write_predictions, even on upstream failure.

Environment variables required (set in Airflow UI or .env):
    BIRDNET_PIPELINE_BUCKET      S3 bucket name
    BIRDNET_PIPELINE_INPUT_PREFIX  S3 prefix for incoming .wav files (e.g. clips/incoming)
    BIRDNET_PIPELINE_OUTPUT_PREFIX S3 prefix for output CSVs (e.g. clips/results)
    BIRDNET_PIPELINE_DYNAMO_TABLE  DynamoDB table name (e.g. birdnet-predictions)
    BIRDNET_PIPELINE_PROJECT_DIR   Absolute path to this repo on the worker
    AWS_DEFAULT_REGION             AWS region (e.g. us-east-1)
"""

import json
import os
from datetime import datetime, timedelta, timezone

import boto3
from airflow import DAG
from airflow.models import Variable
from airflow.operators.python import PythonOperator
from airflow.providers.amazon.aws.sensors.s3 import S3KeySensor

# ---------------------------------------------------------------------------
# Config from environment
# ---------------------------------------------------------------------------
S3_BUCKET = os.environ.get("BIRDNET_PIPELINE_BUCKET", "rainforest-audio")
S3_INPUT_PREFIX = os.environ.get("BIRDNET_PIPELINE_INPUT_PREFIX", "clips/incoming")
S3_OUTPUT_PREFIX = os.environ.get("BIRDNET_PIPELINE_OUTPUT_PREFIX", "clips/results")
DYNAMO_TABLE = os.environ.get("BIRDNET_PIPELINE_DYNAMO_TABLE", "birdnet-predictions")
PROJECT_DIR = os.environ.get("BIRDNET_PIPELINE_PROJECT_DIR", "/opt/rainforest-audio-detection")
AWS_REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")

SCRATCH_BASE = "/tmp/birdnet_pipeline"

# ---------------------------------------------------------------------------
# DAG definition
# ---------------------------------------------------------------------------
default_args = {
    "owner": "airflow",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="rainforest_pipeline",
    description="TinyCNN → BirdNET FP16 batch species detection pipeline",
    default_args=default_args,
    start_date=datetime(2026, 1, 1),
    schedule_interval=None,  # no time-based schedule; s3_sensor task drives execution
    catchup=False,
    tags=["bioacoustics", "birdnet", "tinycnn"],
) as dag:

    # -----------------------------------------------------------------------
    # Task 1: S3KeySensor — wait for new .wav files in the input prefix
    # -----------------------------------------------------------------------
    s3_sensor = S3KeySensor(
        task_id="s3_sensor",
        bucket_name=S3_BUCKET,
        bucket_key=f"{S3_INPUT_PREFIX}/*.wav",
        wildcard_match=True,
        aws_conn_id="aws_default",
        mode="reschedule",          # releases worker slot between pokes
        poke_interval=300,          # check every 5 minutes
        timeout=60 * 60 * 6,        # give up after 6 hours
    )

    # -----------------------------------------------------------------------
    # Task 2: list_new_clips — list .wav files in S3 input prefix
    # -----------------------------------------------------------------------
    def list_new_clips(**context):
        s3 = boto3.client("s3", region_name=AWS_REGION)
        paginator = s3.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=S3_BUCKET, Prefix=S3_INPUT_PREFIX)

        keys = []
        for page in pages:
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if key.endswith(".wav"):
                    keys.append(key)

        if not keys:
            raise ValueError(f"No .wav files found in s3://{S3_BUCKET}/{S3_INPUT_PREFIX}/")

        print(f"Found {len(keys)} clips in s3://{S3_BUCKET}/{S3_INPUT_PREFIX}/")

        scratch_dir = os.path.join(SCRATCH_BASE, context["run_id"])
        os.makedirs(scratch_dir, exist_ok=True)

        manifest = os.path.join(scratch_dir, "clips.json")
        with open(manifest, "w") as f:
            json.dump({"s3_keys": keys, "scratch_dir": scratch_dir}, f)

        return manifest

    list_clips_task = PythonOperator(
        task_id="list_new_clips",
        python_callable=list_new_clips,
    )

    # -----------------------------------------------------------------------
    # Task 3: download_clips — download .wav files from S3 to scratch dir
    # -----------------------------------------------------------------------
    def download_clips(**context):
        manifest_path = context["ti"].xcom_pull(task_ids="list_new_clips")
        with open(manifest_path) as f:
            manifest = json.load(f)

        s3 = boto3.client("s3", region_name=AWS_REGION)
        scratch_dir = manifest["scratch_dir"]
        wav_dir = os.path.join(scratch_dir, "wavs")
        os.makedirs(wav_dir, exist_ok=True)

        local_paths = []
        for key in manifest["s3_keys"]:
            filename = os.path.basename(key)
            local_path = os.path.join(wav_dir, filename)
            s3.download_file(S3_BUCKET, key, local_path)
            local_paths.append(local_path)

        print(f"Downloaded {len(local_paths)} clips to {wav_dir}")

        manifest["local_paths"] = local_paths
        downloaded_manifest = os.path.join(scratch_dir, "clips_downloaded.json")
        with open(downloaded_manifest, "w") as f:
            json.dump(manifest, f)

        return downloaded_manifest

    download_task = PythonOperator(
        task_id="download_clips",
        python_callable=download_clips,
    )

    # -----------------------------------------------------------------------
    # Task 4: tinycnn_filter — run TinyCNN, keep only meaningful clips
    # -----------------------------------------------------------------------
    def tinycnn_filter(**context):
        import sys
        import torch
        import numpy as np
        import soundfile as sf

        sys.path.insert(0, PROJECT_DIR)
        from src.model.architecture import TinyCNN

        manifest_path = context["ti"].xcom_pull(task_ids="download_clips")
        with open(manifest_path) as f:
            manifest = json.load(f)

        scratch_dir = manifest["scratch_dir"]
        ck_path = os.path.join(PROJECT_DIR, "outputs", "models", "tinycnn_v3.pth")
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)

        model = TinyCNN()
        model.load_state_dict(ck["model_state_dict"])
        model.eval()

        SPEC_SR = 22050
        N_MELS = 64
        HOP_LENGTH = 512
        N_FFT = 1024

        import librosa

        # Variable values are stored as JSON strings in the Airflow UI (e.g. "0.5", "5")
        threshold = Variable.get("tinycnn_threshold", default_var=0.5, deserialize_json=True)
        manifest["tinycnn_start_at"] = datetime.now(timezone.utc).isoformat()

        meaningful_paths = []
        for wav_path in manifest["local_paths"]:
            audio, sr = sf.read(wav_path)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if sr != SPEC_SR:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=SPEC_SR)

            mel = librosa.feature.melspectrogram(
                y=audio, sr=SPEC_SR, n_mels=N_MELS,
                n_fft=N_FFT, hop_length=HOP_LENGTH,
            )
            mel_db = librosa.power_to_db(mel, ref=np.max)
            x = torch.from_numpy(mel_db).unsqueeze(0).unsqueeze(0)

            with torch.no_grad():
                prob = torch.sigmoid(model(x).squeeze()).item()

            if prob >= threshold:
                meaningful_paths.append(wav_path)

        manifest["tinycnn_end_at"] = datetime.now(timezone.utc).isoformat()
        print(f"TinyCNN: {len(meaningful_paths)}/{len(manifest['local_paths'])} clips marked meaningful")

        manifest["meaningful_paths"] = meaningful_paths
        with open(manifest_path, "w") as f:
            json.dump(manifest, f)

        return manifest_path

    tinycnn_task = PythonOperator(
        task_id="tinycnn_filter",
        python_callable=tinycnn_filter,
    )

    # -----------------------------------------------------------------------
    # Task 5: birdnet_infer — run BirdNET FP16 on meaningful clips
    # -----------------------------------------------------------------------
    def birdnet_infer(**context):
        import sys
        import numpy as np
        import soundfile as sf
        import tensorflow as tf

        sys.path.insert(0, PROJECT_DIR)

        manifest_path = context["ti"].xcom_pull(task_ids="tinycnn_filter")
        with open(manifest_path) as f:
            manifest = json.load(f)

        meaningful_paths = manifest["meaningful_paths"]
        manifest["birdnet_start_at"] = datetime.now(timezone.utc).isoformat()

        if not meaningful_paths:
            print("No meaningful clips — skipping BirdNET inference.")
            manifest["birdnet_end_at"] = datetime.now(timezone.utc).isoformat()
            manifest["predictions"] = []
            with open(manifest_path, "w") as f:
                json.dump(manifest, f)
            return manifest_path

        model_path = os.path.join(PROJECT_DIR, "outputs", "models", "birdnet_v2.4_fp16.tflite")
        labels_path = os.path.join(PROJECT_DIR, "assets", "labels", "en_us.txt")
        with open(labels_path) as f:
            labels = [line.strip() for line in f]

        interp = tf.lite.Interpreter(model_path=model_path)
        interp.allocate_tensors()
        inp = interp.get_input_details()[0]
        out = interp.get_output_details()[0]

        SR = 48000
        N_SAMPLES = 144000

        top_k = Variable.get("birdnet_top_k", default_var=5, deserialize_json=True)
        predictions = []
        for wav_path in meaningful_paths:
            audio, sr = sf.read(wav_path)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if sr != SR:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=SR)

            if len(audio) < N_SAMPLES:
                audio = np.pad(audio, (0, N_SAMPLES - len(audio)))
            else:
                audio = audio[:N_SAMPLES]

            x = audio.reshape(inp["shape"]).astype(inp["dtype"])
            interp.set_tensor(inp["index"], x)
            interp.invoke()
            scores = interp.get_tensor(out["index"]).flatten()

            top_k_idx = np.argsort(scores)[::-1][:top_k]
            top_k_preds = [{"species": labels[i], "confidence": float(scores[i])} for i in top_k_idx]

            predictions.append({
                "clip_name": os.path.basename(wav_path),
                "top1_species": top_k_preds[0]["species"],
                "top1_confidence": top_k_preds[0]["confidence"],
                "top_k": top_k_preds,
            })

        manifest["birdnet_end_at"] = datetime.now(timezone.utc).isoformat()
        print(f"BirdNET: inferred {len(predictions)} meaningful clips")

        manifest["predictions"] = predictions
        with open(manifest_path, "w") as f:
            json.dump(manifest, f)

        return manifest_path

    birdnet_task = PythonOperator(
        task_id="birdnet_infer",
        python_callable=birdnet_infer,
    )

    # -----------------------------------------------------------------------
    # Task 6: write_predictions — write to DynamoDB + upload CSV to S3
    # -----------------------------------------------------------------------
    def write_predictions(**context):
        import csv

        manifest_path = context["ti"].xcom_pull(task_ids="birdnet_infer")
        with open(manifest_path) as f:
            manifest = json.load(f)

        predictions = manifest["predictions"]
        dag_run_id = context["run_id"]
        processed_at = datetime.now(timezone.utc).isoformat()

        def _duration_s(start_key, end_key):
            try:
                start = datetime.fromisoformat(manifest[start_key])
                end = datetime.fromisoformat(manifest[end_key])
                return round((end - start).total_seconds(), 2)
            except (KeyError, ValueError):
                return None

        dynamo = boto3.resource("dynamodb", region_name=AWS_REGION)
        table = dynamo.Table(DYNAMO_TABLE)

        n_total = len(manifest["local_paths"])
        n_meaningful = len(predictions)

        with table.batch_writer() as batch:
            for pred in predictions:
                batch.put_item(Item={
                    "clip_id": pred["clip_name"],
                    "processed_at": processed_at,
                    "dag_run_id": dag_run_id,
                    "top1_species": pred["top1_species"],
                    "top1_confidence": str(round(pred["top1_confidence"], 4)),
                    "top_k": json.dumps(pred["top_k"]),
                })

        # Batch summary record — one row per DAG run for stage-level reporting
        table.put_item(Item={
            "clip_id": f"BATCH#{dag_run_id}",
            "processed_at": processed_at,
            "dag_run_id": dag_run_id,
            "n_clips_total": n_total,
            "n_meaningful": n_meaningful,
            "n_rejected": n_total - n_meaningful,
            "meaningful_rate": str(round(n_meaningful / n_total, 4)) if n_total > 0 else "0",
            "tinycnn_duration_s": str(_duration_s("tinycnn_start_at", "tinycnn_end_at")),
            "birdnet_duration_s": str(_duration_s("birdnet_start_at", "birdnet_end_at")),
            "dynamo_write_count": str(n_meaningful + 1),
        })

        print(f"Wrote {n_meaningful} clip predictions + 1 batch summary to DynamoDB table '{DYNAMO_TABLE}'")

        scratch_dir = manifest["scratch_dir"]
        csv_path = os.path.join(scratch_dir, "predictions.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["clip_name", "top1_species", "top1_confidence"])
            writer.writeheader()
            for pred in predictions:
                writer.writerow({
                    "clip_name": pred["clip_name"],
                    "top1_species": pred["top1_species"],
                    "top1_confidence": round(pred["top1_confidence"], 4),
                })

        s3 = boto3.client("s3", region_name=AWS_REGION)
        s3_key = f"{S3_OUTPUT_PREFIX}/{dag_run_id}/predictions.csv"
        s3.upload_file(csv_path, S3_BUCKET, s3_key)
        print(f"Uploaded results to s3://{S3_BUCKET}/{s3_key}")

        # Move processed clips out of incoming prefix so sensor won't re-trigger
        for s3_input_key in manifest["s3_keys"]:
            filename = os.path.basename(s3_input_key)
            dest_key = f"clips/processed/{dag_run_id}/{filename}"
            s3.copy_object(
                Bucket=S3_BUCKET,
                CopySource={"Bucket": S3_BUCKET, "Key": s3_input_key},
                Key=dest_key,
            )
            s3.delete_object(Bucket=S3_BUCKET, Key=s3_input_key)
        print(f"Moved {len(manifest['s3_keys'])} clips to clips/processed/{dag_run_id}/")

    write_task = PythonOperator(
        task_id="write_predictions",
        python_callable=write_predictions,
    )

    # -----------------------------------------------------------------------
    # Task 7: cleanup_temp — delete scratch dir (runs even on upstream failure)
    # -----------------------------------------------------------------------
    def cleanup_temp(**context):
        import shutil
        scratch_dir = os.path.join(SCRATCH_BASE, context["run_id"])
        if os.path.exists(scratch_dir):
            shutil.rmtree(scratch_dir)
            print(f"Cleaned up {scratch_dir}")
        else:
            print(f"Scratch dir not found, nothing to clean: {scratch_dir}")

    cleanup_task = PythonOperator(
        task_id="cleanup_temp",
        python_callable=cleanup_temp,
        trigger_rule="all_done",
    )

    # -----------------------------------------------------------------------
    # Dependencies
    # -----------------------------------------------------------------------
    s3_sensor >> list_clips_task >> download_task >> tinycnn_task >> birdnet_task >> write_task >> cleanup_task

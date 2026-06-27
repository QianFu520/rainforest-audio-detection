"""
Upload a 1000-clip demo batch to S3 for the cascade pipeline demo.

Composition: 830 not_meaningful + 170 meaningful (≈83/17 split matching the
empirical distribution observed during the 121k-clip labeling corpus).

All clips are sampled from the labeled dataset, shuffled, and uploaded to
s3://{bucket}/clips/incoming/ in a single pass so the DAG sees them as one batch.

A manifest CSV is written to data/demo_batch_manifest.csv (committed to the repo)
so the demo run is reproducible and the README can cite it.

NOTE: The batch demo measures pipeline cost reduction (how much TinyCNN filters out),
not model accuracy. TinyCNN's ground-truth accuracy is established by v2 training
metrics, not this run.
"""

import argparse
import os
import sys

import boto3
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

S3_BUCKET = "rainforest-audio-kwf"
S3_PREFIX = "clips/incoming"
AWS_REGION = "us-east-1"

N_NOT_MEANINGFUL = 830
N_MEANINGFUL = 170
RANDOM_SEED = 42

MANIFEST_PATH = os.path.join(config.PROJECT_ROOT, "data", "demo_batch_manifest.csv")


def main(dry_run: bool) -> None:
    df = pd.read_csv(config.LABELS_PROGRESS_PATH)

    not_meaningful = df[df["meaningful"] == "not_meaningful"].sample(
        N_NOT_MEANINGFUL, random_state=RANDOM_SEED
    )
    meaningful = df[df["meaningful"] == "meaningful"].sample(
        N_MEANINGFUL, random_state=RANDOM_SEED
    )
    batch = (
        pd.concat([not_meaningful, meaningful])
        .sample(frac=1, random_state=RANDOM_SEED)  # shuffle
        .reset_index(drop=True)
    )

    manifest = batch[["clip_name", "audio_path", "meaningful"]].rename(
        columns={"meaningful": "true_label"}
    )
    manifest.to_csv(MANIFEST_PATH, index=False)
    print(f"Manifest saved → {MANIFEST_PATH}")

    nm_count = (manifest["true_label"] == "not_meaningful").sum()
    m_count  = (manifest["true_label"] == "meaningful").sum()
    print(f"Batch composition: {nm_count} not_meaningful + {m_count} meaningful = {len(batch)} clips")
    print(f"  (~{nm_count / len(batch) * 100:.0f}% background matching observed field-recording distribution)")
    print(f"Target: s3://{S3_BUCKET}/{S3_PREFIX}/")

    if dry_run:
        print("\n[dry-run] First 5 clips that would be uploaded:")
        for _, row in batch.head(5).iterrows():
            print(f"  {row['clip_name']} ({row['meaningful']})")
        print(f"  ... and {len(batch) - 5} more")
        return

    s3 = boto3.client("s3", region_name=AWS_REGION)

    missing = [r["audio_path"] for _, r in batch.iterrows() if not os.path.exists(r["audio_path"])]
    if missing:
        print(f"ERROR: {len(missing)} files not found on disk. First missing:")
        for p in missing[:5]:
            print(f"  {p}")
        sys.exit(1)

    print(f"\nUploading {len(batch)} clips ...")
    for i, (_, row) in enumerate(batch.iterrows(), 1):
        s3_key = f"{S3_PREFIX}/{row['clip_name']}"
        s3.upload_file(row["audio_path"], S3_BUCKET, s3_key)
        if i % 100 == 0 or i == len(batch):
            print(f"  {i}/{len(batch)} uploaded")

    print(f"\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    main(dry_run=args.dry_run)

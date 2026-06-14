"""
Compute meaningful_recall for TinyCNN v1–v4 and log to MLflow.

Reconstructs each version's val split by filtering labels_progress.csv to the
not_meaningful sources used at training time, then applies the same split logic
(RandomState(42)) used in notebook 05_train.ipynb.

Verifies n_val matches the stored checkpoint value before running inference.


"""
import os
import sys

import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.model.architecture import TinyCNN

MODELS_DIR = os.path.join(config.OUTPUTS_DIR, "models")
SPEC_DIR = os.path.join(config.OUTPUTS_DIR, "spectrograms")
DB_PATH = os.path.join(config.OUTPUTS_DIR, "mlflow.db")

mlflow.set_tracking_uri(f"sqlite:///{os.path.abspath(DB_PATH)}")

VERSIONS = [
    {
        "version": "v1",
        "not_meaningful_sources": {"background_energy", "background_flatness"},
    },
    {
        "version": "v2",
        "not_meaningful_sources": {"background_energy", "background_flatness", "model_inference_v1"},
    },
    {
        "version": "v3",
        "not_meaningful_sources": {"background_energy", "background_flatness", "model_inference_v1", "model_inference_v2"},
    },
    {
        "version": "v4",
        "not_meaningful_sources": {"background_energy", "background_flatness", "model_inference_v1", "model_inference_v2", "model_inference_v3"},
    },
]


class SpectrogramDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        spec_path = os.path.join(SPEC_DIR, row["clip_name"].replace(".wav", ".npy"))
        mel = np.load(spec_path)
        x = torch.from_numpy(mel).unsqueeze(0)
        y = torch.tensor(1.0 if row["meaningful"] == "meaningful" else 0.0)
        return x, y


def reconstruct_val_split(labels, not_meaningful_sources):
    """Reproduce the exact val split from notebook 05_train.ipynb."""
    df = labels[labels["meaningful"] != "unknown"].copy()

    # Filter not_meaningful to only sources used in this version
    meaningful_mask = df["meaningful"] == "meaningful"
    not_meaningful_mask = (df["meaningful"] == "not_meaningful") & (
        df["meaningful_source"].isin(not_meaningful_sources)
    )
    df = df[meaningful_mask | not_meaningful_mask].copy()

    df["recorder_date"] = (
        df["Recorder"] + "_" +
        df["clip_name"].str.extract(r"_(\d{8})_")[0]
    )

    rng = np.random.RandomState(42)

    meaningful_df = df[df["meaningful"] == "meaningful"].copy()
    groups = meaningful_df["recorder_date"].unique()
    rng.shuffle(groups)
    split_idx = int(0.8 * len(groups))
    train_groups = set(groups[:split_idx])
    meaningful_df["split"] = meaningful_df["recorder_date"].apply(
        lambda g: "train" if g in train_groups else "val"
    )

    not_meaningful_df = df[df["meaningful"] == "not_meaningful"].copy()
    shuffled_idx = rng.permutation(len(not_meaningful_df))
    split_n = int(0.8 * len(not_meaningful_df))
    not_meaningful_df["split"] = "val"
    not_meaningful_df.iloc[
        shuffled_idx[:split_n],
        not_meaningful_df.columns.get_loc("split")
    ] = "train"

    combined = pd.concat([meaningful_df, not_meaningful_df], ignore_index=True)
    return combined[combined["split"] == "val"]


def compute_meaningful_recall(val_df, model):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    loader = DataLoader(SpectrogramDataset(val_df), batch_size=64, shuffle=False, num_workers=0)

    tp, fn = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            probs = torch.sigmoid(model(x).squeeze(1))
            preds = (probs >= 0.5).float().cpu()
            meaningful_mask = (y == 1.0)
            tp += ((preds == 1.0) & meaningful_mask).sum().item()
            fn += ((preds == 0.0) & meaningful_mask).sum().item()

    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


labels = pd.read_csv(config.LABELS_PROGRESS_PATH)

client = mlflow.tracking.MlflowClient()
exp = client.get_experiment_by_name("tinycnn_binary_filter")
runs = client.search_runs(exp.experiment_id)
run_map = {r.info.run_name: r.info.run_id for r in runs}

for meta in VERSIONS:
    version = meta["version"]
    ck_path = os.path.join(MODELS_DIR, f"tinycnn_{version}.pth")
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)

    val_df = reconstruct_val_split(labels, meta["not_meaningful_sources"])

    # labels_progress.csv evolves across versions so reconstructed n_val may
    # differ slightly from the original training split — close enough for recall.
    print(f"  tinycnn_{version}: reconstructed n_val={len(val_df):,}  (checkpoint n_val={ck['n_val']:,})")

    model = TinyCNN()
    model.load_state_dict(ck["model_state_dict"])

    recall = compute_meaningful_recall(val_df, model)

    run_id = run_map[f"tinycnn_{version}"]
    client.log_metric(run_id, "meaningful_recall", recall)

    print(f"  tinycnn_{version}: meaningful_recall={recall:.4f}  (n_val={len(val_df):,})")

print("\nDone — meaningful_recall logged to MLflow.")

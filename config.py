"""Configuration for rainforest-audio-detection project.
All paths and constants live here so scripts don't hardcode them.
"""

import os

# Project root — the folder this config file lives in
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Data directory
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# Input datasets (not in git — see .gitignore)
EVENT_TABLE_PATH = os.path.join(DATA_DIR, "event_table.csv")
CLIP_CATALOG_PATH = os.path.join(DATA_DIR, "clip_catalog.csv")

# Raw audio lives outside the repo (too large to store with the project)
AUDIO_DIR = "/Users/qian/KWF/Segmented_Foldered"

# Outputs directory (regenerable artifacts, git-ignored)
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")

# Labeling pipeline intermediate state
LABELS_PROGRESS_PATH = os.path.join(OUTPUTS_DIR, "labels_progress.csv")

import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

EVENT_TABLE_PATH = os.path.join(DATA_DIR, "event_table.csv")
CLIP_CATALOG_PATH = os.path.join(DATA_DIR, "clip_catalog.csv")

AUDIO_DIR = "/Users/qian/KWF/Segmented_Foldered"

OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
LABELS_PROGRESS_PATH = os.path.join(OUTPUTS_DIR, "labels_progress.csv")

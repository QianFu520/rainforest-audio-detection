import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

EVENT_TABLE_PATH = os.path.join(DATA_DIR, "event_table.csv")
CLIP_CATALOG_PATH = os.path.join(DATA_DIR, "clip_catalog.csv")

AUDIO_DIR = "/Users/qian/KWF/Segmented_Foldered"

OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
LABELS_PROGRESS_PATH = os.path.join(OUTPUTS_DIR, "labels_progress.csv")

# MLflow tracking server.
# In Docker (DAG tasks): MLFLOW_TRACKING_URI=http://mlflow:5000 is set in docker-compose.yml.
# For local notebook work: export MLFLOW_TRACKING_URI=http://44.212.7.36:5000
# Falls back to local SQLite only if the env var is not set.
MLFLOW_TRACKING_URI = os.environ.get(
    "MLFLOW_TRACKING_URI",
    f"sqlite:///{os.path.join(OUTPUTS_DIR, 'mlflow.db')}",
)

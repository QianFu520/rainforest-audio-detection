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

import os
from pathlib import Path

# --- File Paths ---
# We define the project root relative to this file's location
# Path(__file__) refers to config.py
DATA_DIR = Path("/content/GeoLoc-CVCS/data")

INPUT_TRAIN = DATA_DIR / "metadata" / "places-classification" / "train_val_with_predictions.csv"
INPUT_TEST = DATA_DIR / "metadata" / "places-classification" / "test_with_predictions.csv"

# Output Paths
OUTPUT_DIR = DATA_DIR / "metadata" / "s2-geo-cells"
OUTPUT_TRAIN = OUTPUT_DIR / "train_val_split_geocells.csv"
OUTPUT_TEST = OUTPUT_DIR / "test_geocells.csv"
OUTPUT_CELL_CENTER = OUTPUT_DIR / "cell_center_dataset.csv"

# --- Partitioning Configurations ---
# List of different tau_max values to test for partitioning
PARTITION_CONFIGS = [
    {"name": "config_1", "tau_max": 50},
    {"name": "config_2", "tau_max": 100},
    {"name": "config_3", "tau_max": 200}
]
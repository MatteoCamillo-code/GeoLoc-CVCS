import os
from pathlib import Path

class S2Config:
    
    def __init__(self, project_root: Path = None):
        # --- File Paths ---
        # We define the project root relative to this file's location
        # Path(__file__) refers to config.py
        if project_root is None:
            return
        
        self.DATA_DIR = project_root / "data"

        self.INPUT_TRAIN = self.DATA_DIR / "metadata" / "places-classification" / "train_val_with_predictions.csv"
        self.INPUT_TEST = self.DATA_DIR / "metadata" / "places-classification" / "test_with_predictions.csv"

        # Output Paths
        self.OUTPUT_DIR = self.DATA_DIR / "metadata" / "s2-geo-cells"
        self.OUTPUT_TRAIN = self.OUTPUT_DIR / "train_val_split_geocells.csv"
        self.OUTPUT_TEST = self.OUTPUT_DIR / "test_geocells.csv"
        self.OUTPUT_CELL_CENTER = self.OUTPUT_DIR / "cell_center_dataset.csv"
        self.OUTPUT_CELL_HIERARCHY = self.OUTPUT_DIR / "cell_hierarchy_dataset.csv"

        # --- Partitioning Configurations ---
        # List of different tau_max values to test for partitioning
        self.PARTITION_CONFIGS = [
            {"name": "config_1", "tau_max": 50},
            {"name": "config_2", "tau_max": 100},
            {"name": "config_3", "tau_max": 200}
        ]
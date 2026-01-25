import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image


class OSV_mini(Dataset):
    """
    Worker-safe Dataset module for Windows DataLoader (spawn).
    - Defined in a .py file so workers can import it.
    - Keeps gps in numpy to reduce pickling overhead.
    - Returns labels as a torch Tensor row (shape [3]) instead of Python tuple.
    - Always returns (image, labels, gps) with dummy labels for non-train splits,
      so batch structure is consistent across loaders.
    """

    def __init__(
        self,
        image_root,
        csv_path,
        transform=None,
        split="total",   # "train" | "val" | "total"
        scene="total",   # "urban" | "natural" | "total"
    ):
        self.image_root = image_root
        self.transform = transform
        self.split = split

        df = pd.read_csv(csv_path)

        # ---------------- SPLIT ----------------
        if split == "train":
            df = df[df["is_train"] == 1]
        elif split == "val":
            df = df[df["is_train"] == 0]

        # ---------------- SCENE ----------------
        if scene == "urban":
            df = df[df["predicted_label"] == "Urban"]
        elif scene == "natural":
            df = df[df["predicted_label"] == "Natural"]
        else:
            df = df[df["predicted_label"] != "Indoor"]

        df = df.reset_index(drop=True)

        # ---------------- IMAGES ----------------
        regions = df["region"].astype(str).to_numpy()
        ids = df["id"].astype(str).to_numpy()
        self.image_paths = [os.path.join(image_root, r, f"{i}.jpg") for r, i in zip(regions, ids)]

        # ---------------- GPS ----------------
        # keep as numpy for smaller pickled dataset; convert per item
        self.gps_np = df[["latitude", "longitude"]].to_numpy(dtype="float32")

        # ---------------- LABELS (train only) ----------------
        self.labels = None
        self.label_maps = {}

        label_cols = ["label_config_1", "label_config_2", "label_config_3"]
        label_arrays = []

        for col in label_cols:
            codes, uniques = pd.factorize(df[col])
            label_arrays.append(codes.astype("int64"))
            self.label_maps[col] = uniques

        # shape: [N, 3]
        self.labels = torch.from_numpy(np.stack(label_arrays, axis=1))  # long tensor

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Important: open file inside worker (safe)
        image = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)

        gps = torch.from_numpy(self.gps_np[idx])  # shape [2], float32

        if self.labels is None:
            labels = torch.full((3,), -1, dtype=torch.long)  # dummy labels
        else:
            labels = self.labels[idx]  # shape [3], long

        return image, labels, gps


def seed_worker(worker_id: int):
    """
    Optional: improves reproducibility with multiple workers.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)

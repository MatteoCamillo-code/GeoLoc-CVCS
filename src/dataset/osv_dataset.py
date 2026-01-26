import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms.functional import convert_image_dtype
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
        label_maps=None,
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
        self.label_maps = {} if label_maps is None else label_maps

        label_cols = ["label_config_1", "label_config_2", "label_config_3"]
        label_arrays = []

        for col in label_cols:
            if label_maps is None:
                # Build mapping (only do this ONCE, typically on train)
                codes, uniques = pd.factorize(df[col])
                self.label_maps[col] = uniques
                label_arrays.append(codes.astype("int64"))
            else:
                # Reuse mapping
                uniques = self.label_maps[col]
                # Map df[col] onto the same category ordering as train
                codes = pd.Categorical(df[col], categories=uniques).codes
                # Unseen labels become -1
                label_arrays.append(codes.astype("int64"))

        # shape: [N, 3]
        self.labels = torch.from_numpy(np.stack(label_arrays, axis=1))  # long tensor

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load as PIL Image for compatibility with torchvision transforms
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        # Convert to tensor if not already done by transform
        if not isinstance(img, torch.Tensor):
            img = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        gps = torch.from_numpy(self.gps_np[idx])
        labels = self.labels[idx]
        return img, labels, gps

def seed_worker(worker_id: int):
    """
    Optional: improves reproducibility with multiple workers.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)


def fast_collate(batch):
    """
    Fast collate function for DataLoader.
    Must be defined in a .py module so workers can import it.
    """
    imgs, labels, gps = zip(*batch)
    return torch.stack(imgs, 0), torch.stack(labels, 0), torch.stack(gps, 0)
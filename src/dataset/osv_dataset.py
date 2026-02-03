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
        coarse_label_idx: list[int] = [0],
    ):
        self.image_root = image_root
        self.transform = transform
        self.split = split

        df = pd.read_csv(csv_path)

        # ---------------- SPLIT ----------------
        if "is_train" in df.columns:
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
        ids = df["id"].astype(str).to_numpy()

        # If region column exists, keep old behavior
        if "region" in df.columns:
            regions = df["region"].astype(str).to_numpy()
            self.image_paths = [os.path.join(image_root, r, f"{i}.jpg") for r, i in zip(regions, ids)]
        else:
            # New structure: /data/raw/<dataset>/<nested_id>.jpg
            # Store root path for lazy resolution in __getitem__
            self.image_root_resolved = os.path.abspath(image_root)
            self.image_ids = ids  # Store IDs for lazy path resolution

        # ---------------- GPS ----------------
        # keep as numpy for smaller pickled dataset; convert per item
        self.gps_np = df[["latitude", "longitude"]].to_numpy(dtype="float32")

        # ---------------- LABELS (train only) ----------------
        self.labels = None
        self.label_maps = {} if label_maps is None else label_maps

        label_cols = [f"label_config_{idx + 1}" for idx in coarse_label_idx]
        available_label_cols = [c for c in label_cols if c in df.columns]

        if available_label_cols:
            label_arrays = []
            for col in available_label_cols:
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

            # shape: [N, K]
            self.labels = torch.from_numpy(np.stack(label_arrays, axis=1))  # long tensor
        else:
            # Fallback when label columns are missing in the CSV
            # Keep shape consistent with coarse_label_idx length
            self.labels = torch.zeros((len(df), len(coarse_label_idx)), dtype=torch.long)

    def __len__(self):
        if hasattr(self, 'image_paths'):
            return len(self.image_paths)
        else:
            return len(self.image_ids)

    def __getitem__(self, idx):
        # Resolve image path
        if hasattr(self, 'image_paths'):
            img_path = self.image_paths[idx]
        else:
            # Lazy path resolution for new structure
            img_path = self._resolve_image_path(idx)
        
        # Load as PIL Image for compatibility with torchvision transforms
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        # Convert to tensor if not already done by transform
        if not isinstance(img, torch.Tensor):
            img = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        gps = torch.from_numpy(self.gps_np[idx])
        labels = self.labels[idx]
        return img, labels, gps
    
    def _resolve_image_path(self, idx):
        """Resolve image path for new CSV structure with nested folders."""
        img_id = self.image_ids[idx]
        rel_path = f"{img_id}.jpg" if not img_id.endswith(".jpg") else img_id
        
        # Direct construction - if ID includes dataset folder, this just works
        # Examples: "osv5m/123456" or "mp16_images/a6/e7/2734062435"
        full_path = os.path.join(self.image_root_resolved, rel_path)
        
        # Fast path: if it exists directly, return it (works when dataset folder is in ID)
        if os.path.exists(full_path):
            return full_path
        
        # Fallback: search dataset subfolders (for backward compatibility)
        # This is slower but handles cases where ID doesn't include the folder
        try:
            for entry in os.listdir(self.image_root_resolved):
                subdir = os.path.join(self.image_root_resolved, entry)
                if os.path.isdir(subdir):
                    candidate = os.path.join(subdir, rel_path)
                    if os.path.exists(candidate):
                        return candidate
        except (OSError, PermissionError):
            pass
        
        # Return the direct path (will produce clear error message if image truly missing)
        return full_path

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
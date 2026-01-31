import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path

class ProjectPaths:
    def __init__(self):
        self.root = self._find_project_root()
        # Relative paths
        self.original_train_csv = self.root / "data" / "metadata" / "original-datasets" / "train_mini.csv"
        self.original_test_csv = self.root / "data" / "metadata" / "original-datasets" / "test_mini.csv"
        self.output_train = self.root / "data" / "metadata" / "places-classification" / "train_with_predictions.csv"
        self.output_test = self.root / "data" / "metadata" / "places-classification" / "test_with_predictions.csv"
        # Base image path
        self.base_image_path = "/root/.cache/kagglehub/datasets/josht000/osv-mini-129k/versions/1"

    def _find_project_root(self):
        current = Path.cwd()
        while current != current.parent:
            if (current / 'README.md').exists() and (current / 'data').exists():
                return current
            current = current.parent
        return Path.cwd()

class SceneDataset(Dataset):
    def __init__(self, image_paths, transform):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        try:
            img = Image.open(path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            return img, path
        except:
            return None, path

def collate_fn(batch):
    batch = list(filter(lambda x: x[0] is not None, batch))
    if len(batch) == 0: return torch.tensor([]), []
    return torch.utils.data.dataloader.default_collate(batch)
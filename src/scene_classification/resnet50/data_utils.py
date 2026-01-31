import os
import torch
import kagglehub
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path

class ProjectPaths:
    def __init__(self):
        # Fetch the path directly via kagglehub as requested
        self.base_image_path = kagglehub.dataset_download("josht000/osv-mini-129k")
        self.root = self._find_project_root()
        
        self.original_train_csv = self.root / "data" / "metadata" / "original-datasets" / "train_mini.csv"
        self.original_test_csv = self.root / "data" / "metadata" / "original-datasets" / "test_mini.csv"
        self.output_train = self.root / "data" / "metadata" / "places-classification" / "train_with_predictions.csv"
        self.output_test = self.root / "data" / "metadata" / "places-classification" / "test_with_predictions.csv"

    def _find_project_root(self):
        current = Path.cwd()
        while current != current.parent:
            if (current / 'README.md').exists():
                return current
            current = current.parent
        return Path.cwd()

class SceneDataset(Dataset):
    def __init__(self, image_paths, transform):
        self.image_paths = image_paths
        self.transform = transform
    def __len__(self): return len(self.image_paths)
    def __getitem__(self, idx):
        path = self.image_paths[idx]
        try:
            img = Image.open(path).convert('RGB')
            return self.transform(img), str(path)
        except: return None, str(path)

def collate_fn(batch):
    batch = list(filter(lambda x: x[0] is not None, batch))
    if not batch: return torch.tensor([]), []
    return torch.utils.data.dataloader.default_collate(batch)
from torch.utils.data import Dataset
import pandas as pd
import os
from PIL import Image
import torch
import kagglehub


class OSV_mini(Dataset):
    '''PyTorch Dataset class to handle the OSV5m_mini dataset.'''
    def __init__(self, image_root, csv_path, transform=None):
        self.image_root = image_root
        self.transform = transform

        df = pd.read_csv(csv_path)

        self.image_paths = [
            os.path.join(image_root, row["region"], str(row["id"])+'.jpg')
            for _, row in df.iterrows()
        ]

        self.gps = torch.tensor(
            df[["latitude", "longitude"]].values,
            dtype=torch.float32
        )

        self.region = df["region"].values
        self.sub_region = df["sub-region"].values
        self.city = df["city"].values

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        gps = self.gps[idx]

        if self.transform:
            image = self.transform(image)

        return image, gps


def load_osv_mini(split = "train", transform=None):
    path = kagglehub.dataset_download("josht000/osv-mini-129k")
    path = path+'/osv5m'
    if split == "train":
        print("Loading OSV-mini train split at path:", path)
        return OSV_mini(image_root=path, csv_path=path+'/train_mini.csv', transform=transform)
    elif split == "test":
        print("Loading OSV-mini test split at path:", path)
        return OSV_mini(image_root=path, csv_path=path+'/test_mini.csv', transform=transform)
    else:
        return KeyError("split must be 'train' or 'test'")

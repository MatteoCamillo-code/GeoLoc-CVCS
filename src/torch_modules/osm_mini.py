from torch.utils.data import Dataset
import pandas as pd
import os
from PIL import Image
import torch



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


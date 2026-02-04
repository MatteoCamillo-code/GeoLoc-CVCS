from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from metrics.geospatial import get_predicted_gps, get_weighted_predicted_gps
from .image_classification import ClassifiedImage
from .model_loading import ModelBundle


@dataclass(frozen=True)
class PredictionRow:
    image_name: str
    scene_type: str
    scene_confidence: float
    predicted_lat: float
    predicted_lon: float


class GPSPredictor:
    def __init__(self, device: torch.device, image_size: int = 224, top_k: int = 5, weighted_distance: bool = False):
        self._device = device
        self._top_k = top_k
        self._weighted_distance = weighted_distance
        self._transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

    def _predict_batch(self, model: torch.nn.Module, batch_images: list[torch.Tensor],
                       label_maps: dict, cells_hierarchy, cell_centers, top_k: int, weighted_distance: bool) -> torch.Tensor:
        batch_tensor = torch.stack(batch_images).to(self._device)
        logits = model(batch_tensor)
        # Model returns a list of logits (one per hierarchy level)
        if isinstance(logits, list):
            if weighted_distance:
                return get_weighted_predicted_gps(
                    logits=logits,
                    cells_hierarchy=cells_hierarchy,
                    labels_map=label_maps,
                    top_k=top_k,
                    device=self._device,
                )
            # Use the first level (coarsest) for non-weighted predictions
            logits = logits[0]
        
        return get_predicted_gps(
            predicted_class_indices=torch.argmax(logits, dim=1),
            cell_centers=cell_centers,
            labels_map=label_maps,
            device=self._device,
        )
            

    def predict(self, classified: dict[str, list[ClassifiedImage]], bundle: ModelBundle,
                image_root: Path, batch_size: int = 32) -> pd.DataFrame:
        scenes_to_process: dict[str, list[ClassifiedImage]] = {}
        if "urban" in bundle.models and "natural" in bundle.models:
            scenes_to_process = {
                "urban": classified.get("urban", []),
                "natural": classified.get("natural", []),
            }
        else:
            scene_key = bundle.available_scenes[0]
            scenes_to_process = {
                scene_key: classified.get("urban", []) + classified.get("natural", [])
            }
        
        
        
        rows: list[PredictionRow] = []
        name_to_path: dict[str, Path] = {}
        for img_path in image_root.glob("**/*.jpg"):
            if img_path.name not in name_to_path:
                name_to_path[img_path.name] = img_path

        for scene, images in scenes_to_process.items():
            if not images:
                continue

            model = bundle.models[scene]
            label_maps = bundle.label_maps[scene]
            cells_hierarchy = bundle.cells_hierarchy[scene]
            cell_centers = bundle.cell_centers[scene]

            for i in tqdm(range(0, len(images), batch_size), desc=f"Predicting GPS ({scene})"):
                batch_items = images[i:i + batch_size]
                batch_images: list[torch.Tensor] = []
                valid_items: list[ClassifiedImage] = []

                for item in batch_items:
                    try:
                        img_path = name_to_path.get(item.image_name)
                        if img_path is None:
                            continue
                        img = Image.open(img_path).convert("RGB")
                        batch_images.append(self._transform(img))
                        valid_items.append((item, img_path))
                    except Exception:
                        continue

                if not batch_images:
                    continue

                predicted_gps = self._predict_batch(
                    model=model,
                    batch_images=batch_images,
                    label_maps=label_maps,
                    cells_hierarchy=cells_hierarchy,
                    cell_centers=cell_centers,
                    top_k=self._top_k,
                    weighted_distance=self._weighted_distance,
                )

                for j, (item, img_path) in enumerate(valid_items):
                    pred_lat, pred_lon = predicted_gps[j].cpu().numpy()
                    rows.append(PredictionRow(
                        image_name=item.image_name,
                        scene_type=scene,
                        scene_confidence=float(item.confidence),
                        predicted_lat=float(pred_lat),
                        predicted_lon=float(pred_lon),
                    ))

        return pd.DataFrame([r.__dict__ for r in rows])

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch

from metrics.geospatial import haversine_km, geo_accuracy


@dataclass(frozen=True)
class EvaluationResult:
    merged: pd.DataFrame
    accuracy: dict
    mean_km: float
    median_km: float
    min_km: float
    max_km: float


class Evaluator:
    def evaluate(self, results_df: pd.DataFrame, ground_truth_csv: Path | None) -> EvaluationResult | None:

        if ground_truth_csv is None or not ground_truth_csv.exists():
            return None

        gt_df = pd.read_csv(ground_truth_csv)
        results_df = results_df.copy()

        if "image_name" not in results_df.columns:
            results_df["image_name"] = results_df["image_path"].apply(lambda x: Path(x).stem).astype(str)
        
        # Strip file extensions from results_df image names (e.g., "image.jpg" -> "image")
        results_df["image_name"] = results_df["image_name"].apply(lambda x: Path(x).stem).astype(str)
        gt_df["image_name"] = gt_df["id"].astype(str)

        # Merge with ground truth (inner join for matched images)
        merged = results_df.merge(
            gt_df[["image_name", "latitude", "longitude"]].rename(
                columns={"latitude": "true_latitude", "longitude": "true_longitude"}
            ),
            on="image_name",
            how="inner",
        )

        if merged.empty:
            print("No matching images found between results and ground truth for evaluation.")
            return None

        true_gps = torch.tensor(merged[["true_latitude", "true_longitude"]].values, dtype=torch.float32)
        pred_gps = torch.tensor(merged[["predicted_lat", "predicted_lon"]].values, dtype=torch.float32)

        distances_km = haversine_km(pred_gps, true_gps)
        merged["distance_km"] = distances_km.numpy()

        accuracy = geo_accuracy(distances_km, thresholds=(1, 5, 25, 100))

        # Also add ground truth and distance info to original results_df
        results_df = results_df.merge(
            gt_df[["image_name", "latitude", "longitude"]].rename(
                columns={"latitude": "true_latitude", "longitude": "true_longitude"}
            ),
            on="image_name",
            how="left",
        )
        
        # Add distance column (NaN for unmatched images)
        distance_map = merged.set_index("image_name")["distance_km"].to_dict()
        results_df["distance_km"] = results_df["image_name"].map(distance_map)

        return EvaluationResult(
            merged=merged,
            accuracy=accuracy,
            mean_km=float(distances_km.mean().item()),
            median_km=float(distances_km.median().item()),
            min_km=float(distances_km.min().item()),
            max_km=float(distances_km.max().item()),
        )

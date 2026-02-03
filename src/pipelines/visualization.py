from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt


def plot_predictions(results_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(15, 8))
    scatter = ax.scatter(
        results_df["predicted_lon"],
        results_df["predicted_lat"],
        c=results_df["scene_confidence"],
        cmap="viridis",
        alpha=0.6,
        s=20,
        edgecolors="black",
        linewidth=0.5,
    )
    ax.set_xlabel("Longitude", fontsize=12)
    ax.set_ylabel("Latitude", fontsize=12)
    ax.set_title(f"Predicted GPS Locations (n={len(results_df)})", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Scene Classification Confidence", fontsize=10)
    plt.tight_layout()
    plt.show()


def show_sample_images(results_df: pd.DataFrame, image_root: Path, n_samples: int = 6) -> None:
    n_samples = min(n_samples, len(results_df))
    if n_samples == 0:
        return

    # Map image names (without extension) to file paths
    name_to_path = {}
    for img_path in image_root.glob("**/*.jpg"):
        stem = img_path.stem  # filename without extension
        if stem not in name_to_path:
            name_to_path[stem] = img_path

    sample_indices = np.random.choice(len(results_df), n_samples, replace=False)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, sample_idx in enumerate(sample_indices):
        row = results_df.iloc[sample_idx]
        try:
            img_path = name_to_path.get(row["image_name"])
            if img_path is None:
                raise FileNotFoundError
            img = Image.open(img_path)
            axes[idx].imshow(img)
            axes[idx].axis("off")
            title = f"Pred: ({row['predicted_lat']:.2f}, {row['predicted_lon']:.2f})\n"
            title += f"Confidence: {row['scene_confidence']:.3f}"
            axes[idx].set_title(title, fontsize=9)
        except Exception:
            axes[idx].text(0.5, 0.5, "Error loading image", ha="center", va="center", transform=axes[idx].transAxes)
            axes[idx].axis("off")

    plt.suptitle("Sample Images with GPS Predictions", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()

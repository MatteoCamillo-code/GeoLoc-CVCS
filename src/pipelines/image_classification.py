from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image
from tqdm import tqdm

from models.classifier import SceneClassifierWithConfidence


@dataclass(frozen=True)
class ClassifiedImage:
    image_name: str
    confidence: float


class ImageClassifier:
    def __init__(self, confidence_threshold: float = 0.0):
        self._classifier = SceneClassifierWithConfidence()
        self._threshold = confidence_threshold

    def classify_directory(self, image_root: Path, batch_size: int = 32) -> dict[str, list[ClassifiedImage]]:
        image_files = list(image_root.glob("**/*.jpg"))
        classified: dict[str, list[ClassifiedImage]] = {"urban": [], "natural": []}

        for i in tqdm(range(0, len(image_files), batch_size), desc="Classifying batches"):
            batch_files = image_files[i:i + batch_size]
            batch_images: list[Image.Image] = []
            batch_paths: list[Path] = []

            for img_path in batch_files:
                try:
                    img = Image.open(img_path).convert("RGB")
                except Exception:
                    continue
                batch_images.append(img)
                batch_paths.append(img_path)

            if not batch_images:
                continue

            labels, confidences = self._classifier.process_images(batch_images)

            for path, label_int, conf in zip(batch_paths, labels, confidences):
                label_str = self._classifier.label_int_to_str(label_int)
                if label_str == "Indoor":
                    continue
                if conf < self._threshold:
                    continue

                if label_str == "Urban":
                    classified["urban"].append(ClassifiedImage(image_name=path.name, confidence=conf))
                elif label_str == "Natural":
                    classified["natural"].append(ClassifiedImage(image_name=path.name, confidence=conf))

        return classified

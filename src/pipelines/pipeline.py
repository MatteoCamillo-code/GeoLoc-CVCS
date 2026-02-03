from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch

from .config import PipelineConfig
from .image_classification import ImageClassifier, ClassifiedImage
from .model_loading import ModelLoader, ModelBundle
from .gps_prediction import GPSPredictor
from .evaluation import Evaluator, EvaluationResult


class GeoLocPipeline:
    def __init__(self, project_root: Path, data_dir: Path, config: PipelineConfig):
        self._project_root = project_root
        self._data_dir = data_dir
        self._config = config.resolve_paths()

        device = self._config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._device = torch.device(device)

        self._image_classifier = ImageClassifier(confidence_threshold=self._config.confidence_threshold)
        self._model_loader = ModelLoader(project_root=project_root, data_dir=data_dir, device=self._device)
        self._gps_predictor = GPSPredictor(
            device=self._device,
            image_size=224,
            top_k=self._config.top_k,
        )
        self._evaluator = Evaluator()

    @property
    def device(self) -> torch.device:
        return self._device

    def classify_images(self) -> dict[str, list[ClassifiedImage]]:
        if self._config.image_root is None:
            raise ValueError("PipelineConfig.image_root is required")
        return self._image_classifier.classify_directory(
            image_root=self._config.image_root,
            batch_size=self._config.classify_batch_size,
        )

    def load_models(self) -> ModelBundle:
        checkpoint_path = self._data_dir.parent / "outputs" / "checkpoints" / f"{self._config.checkpoint_name}.pt"
        checkpoint = self._model_loader.load_checkpoint(checkpoint_path)
        scenes = self._model_loader.determine_scenes(checkpoint)

        train_image_root = self._config.train_image_root or (self._data_dir / "raw" / "osv5m")
        return self._model_loader.load_models(
            checkpoint=checkpoint,
            available_scenes=scenes,
            backbone=self._config.backbone,
            use_cbam=self._config.use_cbam,
            cbam_reduction=self._config.cbam_reduction,
            dropout=self._config.dropout,
            coarse_label_idx=self._config.coarse_label_idx,
            image_root_train=train_image_root,
            train_val_csv=self._config.train_val_csv,
        )

    def predict_gps(self, classified: dict[str, list[ClassifiedImage]], bundle: ModelBundle) -> pd.DataFrame:
        if self._config.image_root is None:
            raise ValueError("PipelineConfig.image_root is required")
        return self._gps_predictor.predict(
            classified=classified,
            bundle=bundle,
            image_root=self._config.image_root,
            batch_size=self._config.inference_batch_size,
        )

    def evaluate(self, results_df: pd.DataFrame) -> tuple[EvaluationResult | None, pd.DataFrame]:
        """Evaluate predictions and update results_df with ground truth and distances.
        
        Returns:
            Tuple of (evaluation_result, updated_results_df)
        """
        eval_result = self._evaluator.evaluate(results_df, self._config.ground_truth_csv)
        # The evaluator modifies results_df in place with true coords and distances
        # We need to get the updated version
        if eval_result is not None:
            # Merge ground truth into results_df
            gt_df = pd.read_csv(self._config.ground_truth_csv)
            results_df = results_df.copy()
            
            if "image_name" not in results_df.columns:
                results_df["image_name"] = results_df["image_path"].apply(lambda x: Path(x).stem).astype(str)
            
            # Strip file extensions from results_df image names (e.g., "image.jpg" -> "image")
            results_df["image_name"] = results_df["image_name"].apply(lambda x: Path(x).stem).astype(str)
            gt_df["image_name"] = gt_df["id"].astype(str)
            
            results_df = results_df.merge(
                gt_df[["image_name", "latitude", "longitude"]].rename(
                    columns={"latitude": "true_latitude", "longitude": "true_longitude"}
                ),
                on="image_name",
                how="left",
            )
            
            # Add distance column
            distance_map = eval_result.merged.set_index("image_name")["distance_km"].to_dict()
            results_df["distance_km"] = results_df["image_name"].map(distance_map)
        
        return eval_result, results_df

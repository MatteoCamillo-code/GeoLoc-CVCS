from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional


@dataclass(frozen=True)
class PipelineConfig:
    checkpoint_name: str
    backbone: str = "resnet50"  # "resnet50" | "inceptionv4"
    use_cbam: bool = True
    cbam_reduction: int = 16
    dropout: float = 0.0
    coarse_label_idx: list[int] = field(default_factory=lambda: [0,])
    expanded_dataset: bool = False  # whether to use expanded metadata files (with _expanded suffix)

    # I/O
    image_root: Optional[Path] = None
    ground_truth_csv: Optional[Path] = None
    train_image_root: Optional[Path] = None
    train_val_csv: Optional[Path] = None

    # Runtime
    device: Optional[str] = None
    classify_batch_size: int = 32
    inference_batch_size: int = 32
    confidence_threshold: float = 0.0
    top_k: int = 5
    weighted_distance: bool = True
    
    def resolve_path(self, path: Optional[Path]) -> Optional[Path]:
        return path.resolve() if path is not None else None

    def resolve_paths(self) -> "PipelineConfig":
        return PipelineConfig(
            checkpoint_name=self.checkpoint_name,
            backbone=self.backbone,
            use_cbam=self.use_cbam,
            cbam_reduction=self.cbam_reduction,
            dropout=self.dropout,
            coarse_label_idx=list(self.coarse_label_idx),
            image_root=self.resolve_path(self.image_root),
            ground_truth_csv=self.resolve_path(self.ground_truth_csv),
            train_image_root=self.resolve_path(self.train_image_root),
            train_val_csv=self.resolve_path(self.train_val_csv),
            device=self.device,
            classify_batch_size=self.classify_batch_size,
            inference_batch_size=self.inference_batch_size,
            confidence_threshold=self.confidence_threshold,
            top_k=self.top_k,
            weighted_distance=self.weighted_distance,
            expanded_dataset=self.expanded_dataset,
        )

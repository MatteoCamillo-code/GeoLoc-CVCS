from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import timm
from torchvision.models import resnet50, ResNet50_Weights

from dataset.dataloader_utils import create_dataloaders
from models.multi_head_classifier import MultiHeadClassifier
from models.multi_head_classifier_cbam import MultiHeadClassifier as MultiHeadClassifierCbam


@dataclass(frozen=True)
class ModelBundle:
    models: dict[str, torch.nn.Module]
    label_maps: dict[str, dict]
    cell_centers: dict[str, pd.DataFrame]
    cells_hierarchy: dict[str, pd.DataFrame]
    available_scenes: list[str]


class ModelLoader:
    def __init__(self, project_root: Path, data_dir: Path, device: torch.device):
        self._project_root = project_root
        self._data_dir = data_dir
        self._device = device

    def load_checkpoint(self, checkpoint_path: Path) -> dict:
        return torch.load(checkpoint_path, map_location=self._device, weights_only=False)

    def determine_scenes(self, checkpoint: dict) -> list[str]:
        if "urban" in checkpoint and "natural" in checkpoint:
            return ["urban", "natural"]
        if "total" in checkpoint:
            return ["total"]
        if "model" in checkpoint:
            return ["total"]
        possible = [k for k in checkpoint.keys() if isinstance(checkpoint[k], dict) and "model" in checkpoint[k]]
        if possible:
            return list(possible)
        raise ValueError(f"Could not determine scene structure in checkpoint. Keys: {list(checkpoint.keys())}")

    def _create_backbone_cbam(self, backbone_name: str) -> tuple[nn.Module, int]:
        if backbone_name == "inceptionv4":
            backbone = timm.create_model(
                "inception_v4",
                pretrained=True,
                features_only=True,
                out_indices=(4,),
            )
            feat_dim = 1536
        elif backbone_name == "resnet50":
            resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            backbone = nn.Sequential(*list(resnet.children())[:-2])
            feat_dim = 2048
        else:
            raise ValueError(f"Unsupported backbone type: {backbone_name}")
        return backbone, feat_dim

    def _create_backbone_flat(self, backbone_name: str) -> tuple[nn.Module, int]:
        if backbone_name == "inceptionv4":
            backbone = timm.create_model("inception_v4", pretrained=True)
            feat_dim = 1536
        elif backbone_name == "resnet50":
            weights = ResNet50_Weights.IMAGENET1K_V2
            backbone = resnet50(weights=weights)
            feat_dim = 2048
        else:
            raise ValueError(f"Unsupported backbone type: {backbone_name}")
        backbone = nn.Sequential(*list(backbone.children())[:-1], nn.Flatten(1))
        return backbone, feat_dim

    def _load_label_maps(self, checkpoint: dict, scene: str, coarse_label_idx: list[int],
                         image_root_train: Path, train_val_csv: Path | None) -> dict:
        print(f"Loading label maps for scene '{scene}'...")
        if scene in checkpoint and "label_maps" in checkpoint[scene]:
            return checkpoint[scene]["label_maps"]
        if "label_maps" in checkpoint:
            return checkpoint["label_maps"]

        print(f"Loading label maps for scene '{scene}' from training data...")
        csv_path = train_val_csv
        if csv_path is None or not csv_path.exists():
            csv_path = self._data_dir / "metadata" / "s2-geo-cells" / f"train_val_split_geocells_{scene}.csv"
            if not csv_path.exists():
                csv_path = self._data_dir / "metadata" / "s2-geo-cells" / "train_val_split_geocells_total.csv"

        loader_dict = create_dataloaders(
            image_root=image_root_train,
            csv_path=csv_path,
            batch_size=8,
            num_workers=0,
            img_size=224,
            scenes=[scene],
            train_subset_pct=1.0,
            val_subset_pct=1.0,
            augment=False,
            coarse_label_idx=coarse_label_idx,
        )
        return loader_dict[scene]["label_maps"]

    def load_models(
        self,
        checkpoint: dict,
        available_scenes: list[str],
        backbone: str,
        use_cbam: bool,
        cbam_reduction: int,
        dropout: float,
        coarse_label_idx: list[int],
        image_root_train: Path,
        train_val_csv: Path | None,
        expanded_dataset: bool = False,
    ) -> ModelBundle:
        models: dict[str, torch.nn.Module] = {}
        label_maps: dict[str, dict] = {}
        cell_centers: dict[str, pd.DataFrame] = {}
        cells_hierarchy: dict[str, pd.DataFrame] = {}

        for scene in available_scenes:
            centers_path = self._data_dir / "metadata" / "s2-geo-cells" / f"cell_center_dataset_{scene}{'_expanded' if expanded_dataset else ''}.csv"
            hierarchy_path = self._data_dir / "metadata" / "s2-geo-cells" / f"cell_hierarchy_dataset_{scene}{'_expanded' if expanded_dataset else ''}.csv"
            cell_centers[scene] = pd.read_csv(centers_path, index_col=0)
            cells_hierarchy[scene] = pd.read_csv(hierarchy_path)

            label_maps[scene] = self._load_label_maps(
                checkpoint,
                scene,
                coarse_label_idx,
                image_root_train=image_root_train,
                train_val_csv=train_val_csv,
            )

            if use_cbam:
                backbone_model, feat_dim = self._create_backbone_cbam(backbone)
                model = MultiHeadClassifierCbam(
                    backbone=backbone_model.to(self._device),
                    feat_dim=feat_dim,
                    head_dims=[len(label_maps[scene][f"label_config_{idx + 1}"]) for idx in coarse_label_idx],
                    dropout=dropout,
                    coarse_level_idx=coarse_label_idx,
                    use_cbam=True,
                    cbam_reduction=cbam_reduction,
                ).to(self._device)
            else:
                backbone_model, feat_dim = self._create_backbone_flat(backbone)
                model = MultiHeadClassifier(
                    backbone=backbone_model.to(self._device),
                    feat_dim=feat_dim,
                    head_dims=[len(label_maps[scene][f"label_config_{idx + 1}"]) for idx in coarse_label_idx],
                    dropout=dropout,
                    coarse_level_idx=coarse_label_idx,
                ).to(self._device)

            if scene in checkpoint:
                model.load_state_dict(checkpoint[scene]["model"])
            elif "model" in checkpoint and len(available_scenes) == 1:
                model.load_state_dict(checkpoint["model"])
            else:
                raise ValueError(f"Could not find weights for scene '{scene}' in checkpoint")

            model.eval()
            models[scene] = model

        return ModelBundle(
            models=models,
            label_maps=label_maps,
            cell_centers=cell_centers,
            cells_hierarchy=cells_hierarchy,
            available_scenes=available_scenes,
        )

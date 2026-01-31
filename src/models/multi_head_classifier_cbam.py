import torch
import torch.nn as nn
from .cbam import CBAM


class MultiHeadClassifier(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        feat_dim: int,
        head_dims: list[int],
        dropout: float = 0.0,
        coarse_level_idx: list[int] = [0],
        use_cbam: bool = False,
        cbam_reduction: int = 16,
    ):
        super().__init__()
        
        self.backbone = backbone
        self.use_cbam = use_cbam

        if use_cbam:
            self.cbam = CBAM(feat_dim, cbam_reduction)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

        self.heads = nn.ModuleList(
            [nn.Linear(feat_dim, c) for c in head_dims]
        )
        self.coarse_level_idx = coarse_level_idx

        self.freeze_backbone()

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def forward(self, x):
        feats = self.backbone(x)

        # timm features_only returns a list
        if isinstance(feats, (list, tuple)):
            feats = feats[0]

        # feats: [B, C, H, W]
        if self.use_cbam:
            feats = self.cbam(feats)

        feats = self.pool(feats).flatten(1)

        if self.dropout is not None:
            feats = self.dropout(feats)

        logits = [head(feats) for head in self.heads]
        return logits

    def get_coarse_level_logits(self, x):
        logits = self.forward(x)
        return [logits[idx] for idx in self.coarse_level_idx]

    def get_head_dims(self):
        return [head.out_features for head in self.heads]

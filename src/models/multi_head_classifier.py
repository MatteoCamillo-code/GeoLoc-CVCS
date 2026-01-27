import torch
import torch.nn as nn

class MultiHeadClassifier(nn.Module):
    def __init__(self, backbone: nn.Module, 
                 feat_dim: int, 
                 head_dims: list[int], 
                 dropout: float = 0.0, 
                 coarse_level_idx: list[int] = [0],
                 ):
        super().__init__()
        self.backbone = backbone
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.heads = nn.ModuleList([nn.Linear(feat_dim, c) for c in head_dims])
        self.coarse_level_idx = coarse_level_idx
        
        self.freeze_backbone()
        
    def freeze_backbone(self):
        # Freeze all parameters in the model initially
        for param in self.backbone.parameters():
            param.requires_grad = False
            
    def forward(self, x):
        # backbone should output a feature vector [B, feat_dim]
        feats = self.backbone(x)
        if self.dropout is not None:
            feats = self.dropout(feats)
        logits = [head(feats) for head in self.heads]   # list of [B, Ci]
        return logits

    def get_coarse_level_logits(self, x):
        logits = self.forward(x)
        coarse_logits = [logits[idx] for idx in range(len(self.coarse_level_idx))]
        return coarse_logits
    
    def get_head_dims(self):
        return [head.out_features for head in self.heads]
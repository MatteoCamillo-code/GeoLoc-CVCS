from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossEntropyWithLabelSmoothing(nn.Module):
    def __init__(self, smoothing: float = 0.0, ignore_index: int = -1):
        super().__init__()
        self.smoothing = float(smoothing)
        self.ignore_index = int(ignore_index)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # logits: [B,C], target: [B]
        if self.smoothing <= 0.0:
            return F.cross_entropy(logits, target, ignore_index=self.ignore_index)

        # mask ignored targets
        mask = target != self.ignore_index
        if mask.sum() == 0:
            return logits.sum() * 0.0

        logits = logits[mask]
        target = target[mask]

        n_classes = logits.size(-1)
        log_probs = F.log_softmax(logits, dim=-1)

        # smoothed target distribution
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (n_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)

        return torch.mean(torch.sum(-true_dist * log_probs, dim=-1))

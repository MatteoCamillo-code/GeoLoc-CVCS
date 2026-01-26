import torch
from typing import Optional

@torch.no_grad()
def accuracy_top1(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: Optional[int] = None,
) -> torch.Tensor:
    """
    Returns a scalar tensor (on the same device) with value in [0, 1].
    When all targets are ignored, returns a zero tensor.
    """
    preds = logits.argmax(dim=1)

    if ignore_index is not None:
        mask = targets != ignore_index
        if mask.sum() == 0:
            # return tensor on correct device/dtype
            return logits.new_zeros(())
        preds = preds[mask]
        targets = targets[mask]

    return (preds == targets).float().mean()

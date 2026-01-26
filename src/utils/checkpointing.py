from pathlib import Path
from typing import Optional
import torch


def save_checkpoint(path: str, model, optimizer=None, epoch: int = 0, extra: Optional[dict] = None):
    """Persist model (and optional optimizer) state.

    path: destination file path for the checkpoint (directories are created).
    model: torch.nn.Module with a state_dict() method.
    optimizer: optional torch optimizer to persist its state_dict.
    epoch: training epoch index to store for resuming.
    extra: optional dict for any additional metadata you want to save.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = {"epoch": epoch, "model": model.state_dict()}
    if optimizer is not None:
        # e.g., momentum buffers, running averages
        payload["optimizer"] = optimizer.state_dict()
    if extra:
        payload["extra"] = extra
    torch.save(payload, path)

def load_checkpoint(path: str, model, optimizer=None, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt

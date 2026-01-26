import os, random
import numpy as np
import torch


def seed_everything(seed: int = 42) -> None:
    """Seed Python, NumPy, and PyTorch RNGs; set CUDA/CUDNN flags.

    seed: integer used for all RNGs. Note: cudnn.deterministic=False and
    cudnn.benchmark=True favor speed over bit-exact reproducibility; set
    deterministic=True and benchmark=False if strict determinism is required.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    
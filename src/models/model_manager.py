import torch

def load_torch_model(weight_path, map_location="cpu"):
    """
    Load a PyTorch model or state_dict from the specified path.
    """
    return torch.load(weight_path, map_location=map_location)
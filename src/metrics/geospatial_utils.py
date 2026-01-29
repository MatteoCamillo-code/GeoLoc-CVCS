import torch

def inverse_spheric_transformation(x, y, z):
    """
    Convert 3D Cartesian coordinates back to latitude and longitude.
    Assumes x, y, z are averaged (float) coordinates.
    """
    import math
    r = math.sqrt(x * x + y * y + z * z)
    if r == 0:
        return 0.0, 0.0
    lat = math.asin(z / r) * (180.0 / math.pi)
    lon = math.atan2(y, x) * (180.0 / math.pi)
    return lat, lon

def precompute_class_indices(cells_hierarchy, labels_map, num_levels, device):
    """Pre-compute class indices for all paths and levels."""
    num_paths = len(cells_hierarchy)
    
    if isinstance(labels_map, dict):
        label_maps_by_level = [labels_map.get(f"label_config_{level + 1}") for level in range(num_levels)]
    else:
        label_maps_by_level = list(labels_map)
    
    class_indices = torch.full((num_paths, num_levels), -1, dtype=torch.long, device=device)
    for level in range(num_levels):
        level_map = label_maps_by_level[level]
        col_name = f'config_{level + 1}'
        cell_ids = cells_hierarchy[col_name].values
        for path_idx, cell_id in enumerate(cell_ids):
            try:
                class_idx = level_map.get_loc(cell_id)
                class_indices[path_idx, level] = class_idx
            except (KeyError, AttributeError):
                pass  # Keep -1 for missing
    
    return class_indices

def preload_coordinates(cells_hierarchy, device):
    """Pre-load coordinate tensors to GPU."""
    coord_x = torch.tensor(cells_hierarchy['config_1_x'].values, dtype=torch.float32, device=device)
    coord_y = torch.tensor(cells_hierarchy['config_1_y'].values, dtype=torch.float32, device=device)
    coord_z = torch.tensor(cells_hierarchy['config_1_z'].values, dtype=torch.float32, device=device)
    coord_points = torch.tensor(cells_hierarchy['config_1_points'].values, dtype=torch.float32, device=device)
    return coord_x, coord_y, coord_z, coord_points

def compute_path_scores_for_sample(probs, class_indices, num_paths, num_levels, device):
    """Compute normalized path scores for a single sample."""
    path_scores = torch.zeros((num_paths, num_levels), dtype=torch.float32, device=device)
    for level in range(num_levels):
        valid = class_indices[:, level] >= 0
        if valid.any():
            path_scores[valid, level] = probs[level][class_indices[valid, level]]
    
    col_sums = path_scores.sum(dim=0, keepdim=True)
    col_sums = torch.where(col_sums == 0, torch.ones_like(col_sums), col_sums)
    norm_scores = path_scores / col_sums
    path_sum_scores = norm_scores.sum(dim=1)
    
    return path_sum_scores

def get_topk_gps(topk_indices, coord_x, coord_y, coord_z, coord_points):
    """Get weighted average lat/lon from top-k paths."""
    topk_x = coord_x[topk_indices]
    topk_y = coord_y[topk_indices]
    topk_z = coord_z[topk_indices]
    topk_points = coord_points[topk_indices]
    
    total_points = topk_points.sum()
    if total_points == 0:
        return None
    
    x = (topk_x * topk_points).sum() / total_points
    y = (topk_y * topk_points).sum() / total_points
    z = (topk_z * topk_points).sum() / total_points
    
    lat, lon = inverse_spheric_transformation(x.item(), y.item(), z.item())
    return [lat, lon]
import torch
import math
import metrics.geospatial_utils as utils

@torch.no_grad()
def haversine_km(latlon1: torch.Tensor, latlon2: torch.Tensor) -> torch.Tensor:
    """
    latlon1, latlon2: [B,2] in degrees (lat, lon)
    returns: [B] distance in km
    """
    lat1 = torch.deg2rad(latlon1[:, 0])
    lon1 = torch.deg2rad(latlon1[:, 1])
    lat2 = torch.deg2rad(latlon2[:, 0])
    lon2 = torch.deg2rad(latlon2[:, 1])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # Haversine great-circle distance on a sphere.
    a = torch.sin(dlat/2)**2 + torch.cos(lat1)*torch.cos(lat2)*torch.sin(dlon/2)**2
    c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1-a))
    R = 6371.0088  # mean Earth radius in km; scales radians to kilometers
    return R * c

@torch.no_grad()
def geo_accuracy(dist_km: torch.Tensor, thresholds=(1,5,25,100)) -> dict:
    """Return accuracy at multiple distance thresholds (km) as percentages."""
    return {f"acc@{t}km": f"{round((dist_km <= t).float().mean().item() * 100, 2):.2f}%" for t in thresholds}

def get_predicted_gps(predicted_class_indices, cell_centers, labels_map, device):
    # TODO: generalize for any level
    predicted_s2_cells = labels_map.get("label_config_1")[predicted_class_indices]
    predicted_latlons_df = cell_centers.loc[predicted_s2_cells, ['center_latitude', 'center_longitude']]
    return torch.tensor(predicted_latlons_df.values, dtype=torch.float32, device=device)

def get_weighted_predicted_gps(logits, cells_hierarchy, labels_map, top_k, device):
    """
    Vectorized GPU-accelerated GPS prediction using hierarchical scoring.
    Pre-computes indices and coordinates once, then processes per sample.
    """
    batch_size = logits[0].size(0)
    num_levels = len(logits)
    num_paths = len(cells_hierarchy)
    
    if num_paths == 0:
        return torch.zeros((batch_size, 2), dtype=torch.float32, device=device)
    
    # Pre-compute once outside loop
    class_indices = utils.precompute_class_indices(cells_hierarchy, labels_map, num_levels, device)
    coord_x, coord_y, coord_z, coord_points = utils.preload_coordinates(cells_hierarchy, device)
    
    predicted_latlons = []
    
    for sample_idx in range(batch_size):
        # Get softmax probabilities for each hierarchy level
        probs = [torch.softmax(logits[level][sample_idx], dim=0).float() for level in range(num_levels)]
        
        # Compute normalized path scores
        path_sum_scores = utils.compute_path_scores_for_sample(probs, class_indices, num_paths, num_levels, device)
        
        # Get top-k paths
        k = min(top_k, num_paths)
        _, topk_indices = torch.topk(path_sum_scores, k=k, largest=True)
        
        # Compute weighted GPS from top-k
        gps = utils.get_topk_gps(topk_indices, coord_x, coord_y, coord_z, coord_points)
        if gps is None:
            predicted_latlons.append([0.0, 0.0])
        else:
            predicted_latlons.append(gps)
            
    # Convert to tensor [B, 2] on device
    predicted_gps = torch.tensor(predicted_latlons, dtype=torch.float32, device=device)
    return predicted_gps
import torch
import math

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
    """Return accuracy at multiple distance thresholds (km)."""
    return {f"acc@{t}km": (dist_km <= t).float().mean().item() for t in thresholds}

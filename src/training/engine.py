import torch
from torch.cuda.amp import autocast
from metrics.classification import accuracy_top1
from metrics.geospatial import haversine_km, geo_accuracy

def _to_device(batch, device):
    # dataset.osv_dataset.OSV_mini returns: (image, labels, gps)
    # labels shape: [B, 3] for 3 label configs
    x, labels, gps = batch
    return x.to(device, non_blocking=True), labels.to(device, non_blocking=True), gps.to(device, non_blocking=True) if gps is not None else None

def _compute_logits_and_loss(model, x, labels, criterion):
    logits = model.get_coarse_level_logits(x)
    loss = torch.stack([criterion(logit, labels[:, idx])
                        for idx, logit in enumerate(logits)]).mean()
    return logits, loss

def _compute_accuracy(logits, labels):
    return torch.stack([(logit.argmax(1) == labels[:, idx]).float().mean()
                        for idx, logit in enumerate(logits)]).mean()

def _update_running_metric(total, value, batch_size):
    total += value.detach() * batch_size
    return total

def _get_predicted_gps(logits, cells_hierarchy, labels_map, cell_centers, device):
    """
    Get predicted lat/lon using hierarchical scoring from equation (3).
    
    For each sample and each hierarchy path (row in cells_hierarchy):
    s(path) = sum_{i=1}^{N} [ geoscore(cell_i; C_i) / sum_{t=1}^{K} geoscore(g_t; C_i) ]
    
    where N=3 (hierarchy levels), and we score each cell at each level in the path.
    Selects the path with maximum cumulative score.
    """
    batch_size = logits[0].size(0)
    predicted_latlons = []
    
    for sample_idx in range(batch_size):
        # Get softmax probabilities for each hierarchy level
        probs = [torch.softmax(logits[level][sample_idx], dim=0) for level in range(len(logits))]
        
        # Dictionary to store: cell_id -> cumulative score
        best_score = -float('inf')
        best_cell = None
        
        # Iterate through each hierarchy path in cells_hierarchy
        for _, row in cells_hierarchy.iterrows():
            # Get cells at each level for this hierarchy path
            path_cells = [row[f'config_{level + 1}'] for level in range(len(logits))]
            
            # Compute s(path) by normalizing and summing across all levels
            path_score = 0.0
            for level in range(len(logits)):
                try:
                    # Get class index for this cell at this level
                    cell_id = path_cells[level]
                    class_idx = labels_map[level].index.get_loc(cell_id)
                    
                    # Get probability and normalize
                    geoscore = probs[level][class_idx].item()
                    normalization = probs[level].sum().item()
                    path_score += geoscore / normalization
                except (KeyError, AttributeError):
                    # Cell not in this level's label map, skip
                    path_score = -float('inf')
                    break
            
            # Track best path
            if path_score > best_score:
                best_score = path_score
                best_cell = row['config_1']  # Use finest-level cell
        
        # Look up GPS coordinates for the best cell
        try:
            if best_cell is None:
                lat_lon = [0.0, 0.0]
            else:
                lat_lon = cell_centers.loc[best_cell, ['center_latitude', 'center_longitude']].values
            predicted_latlons.append(lat_lon)
        except KeyError:
            predicted_latlons.append([0.0, 0.0])
    
    # Convert to tensor [B, 2] on device
    predicted_gps = torch.tensor(predicted_latlons, dtype=torch.float32, device=device)
    return predicted_gps

def _finalize_epoch_metrics(total_loss, total_acc, n):
    loss_avg = (total_loss / max(n, 1)).item()
    acc_avg = (total_acc / max(n, 1)).item()
    return loss_avg, acc_avg

def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    scaler,
    device: torch.device,
    amp: bool = True,
    use_tqdm: bool = True,
):
    model.train()
    total_loss = torch.zeros((), device=device)
    total_acc  = torch.zeros((), device=device)
    n = 0

    it = _get_pbar(loader, desc="train", enable=use_tqdm)

    for step, batch in enumerate(it):
        x, labels, gps = _to_device(batch, device)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
            logits, loss = _compute_logits_and_loss(model, x, labels, criterion)
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        acc = _compute_accuracy(logits, labels)

        bs = x.size(0)
        total_loss = _update_running_metric(total_loss, loss, bs)  # tensor, stays on GPU
        total_acc = _update_running_metric(total_acc, acc, bs)      # make sure acc is a tensor
        n += bs

        if use_tqdm and (step % 40 == 0):
            it.set_postfix(
                loss=f"{loss.detach().float().item():.4f}",
                acc=f"{(acc.detach().float().item()*100):.2f}%",
            )   
    loss_avg, acc_avg = _finalize_epoch_metrics(total_loss, total_acc, n)
    return {"loss": loss_avg, "acc": acc_avg}

@torch.inference_mode()
def evaluate(
    model,
    loader,
    cell_centers,
    cells_hierarchy,
    labels_map,
    criterion,
    device: torch.device,
    amp: bool = True,
    use_tqdm: bool = True,
):
    model.eval()
    total_loss = torch.zeros((), device=device)
    total_acc = torch.zeros((), device=device)
    predicted_gps = []  # Use Python list
    true_gps = []       # Use Python list
    n = 0

    it = _get_pbar(loader, desc="val", enable=use_tqdm)

    for step, batch in enumerate(it):
        x, labels, gps = _to_device(batch, device)

        with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
            logits, loss = _compute_logits_and_loss(model, x, labels, criterion)
            predicted_class_indices = logits[0].argmax(dim=1).cpu().numpy()

        acc = _compute_accuracy(logits, labels)
        
        # Collect GPS predictions and ground truth (keep on GPU)
        predicted_gps.append(
            _get_predicted_gps(logits, cells_hierarchy, labels_map, cell_centers, device)
        )
        true_gps.append(gps)
        
        bs = x.size(0)
        total_loss = _update_running_metric(total_loss, loss, bs)  # tensor, stays on GPU
        total_acc = _update_running_metric(total_acc, acc, bs)      # make sure acc is a tensor
        n += bs

        if use_tqdm and (step % 40 == 0):
            it.set_postfix(
                loss=f"{loss.detach().float().item():.4f}",
                acc=f"{(acc.detach().float().item()*100):.2f}%",
            )
    
    # Concatenate all batches (still on GPU)
    predicted_gps = torch.cat(predicted_gps, dim=0)
    true_gps = torch.cat(true_gps, dim=0)
    
    # compute the distances
    distances = haversine_km(predicted_gps, true_gps)
    
    # Calculate geo_accuracy
    geo_acc_avg = geo_accuracy(distances)
    
    loss_avg, acc_avg = _finalize_epoch_metrics(total_loss, total_acc, n)
    
    return {"loss": loss_avg, "acc": acc_avg, "geo_acc": geo_acc_avg}


def _get_pbar(loader, desc: str, enable: bool = True):
    if not enable:
        return loader
    try:
        from tqdm.auto import tqdm
        return tqdm(loader, desc=desc, leave=False)
    except Exception:
        return loader

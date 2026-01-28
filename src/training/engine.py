import torch
from torch.cuda.amp import autocast
from metrics.classification import accuracy_top1
from metrics.geospatial import haversine_km, geo_accuracy

def _to_device(batch, device):
    # dataset.osv_dataset.OSV_mini returns: (image, labels, gps)
    # labels shape: [B, 3] for 3 label configs
    x, labels, gps = batch
    return x.to(device, non_blocking=True), labels.to(device, non_blocking=True), gps.to(device, non_blocking=True) if gps is not None else None

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
            logits = model.get_coarse_level_logits(x)
            loss = torch.stack([criterion(logit, labels[:, idx]) 
                                for idx, logit in enumerate(logits)]).mean()
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        acc = torch.stack([(logit.argmax(1) == labels[:, idx]).float().mean() 
                           for idx, logit in enumerate(logits)]).mean()

        bs = x.size(0)
        total_loss += loss.detach() * bs              # tensor, stays on GPU
        total_acc  += acc.detach() * bs               # make sure acc is a tensor
        n += bs

        if use_tqdm and (step % 40 == 0):
            it.set_postfix(
                loss=f"{loss.detach().float().item():.4f}",
                acc=f"{(acc.detach().float().item()*100):.2f}%",
            )   
    loss_avg = (total_loss / max(n, 1)).item()
    acc_avg  = (total_acc  / max(n, 1)).item()
    return {"loss": loss_avg, "acc": acc_avg}

@torch.inference_mode()
def evaluate(
    model,
    loader,
    cell_centers,
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
            logits = model.get_coarse_level_logits(x)
            loss = torch.stack([criterion(logit, labels[:, idx]) 
                                for idx, logit in enumerate(logits)]).mean()
            predicted_class_indices = logits[0].argmax(dim=1).cpu().numpy()

        acc = torch.stack([(logit.argmax(1) == labels[:, idx]).float().mean() 
                           for idx, logit in enumerate(logits)]).mean()
        
        # Collect GPS predictions and ground truth (keep on GPU)
        predicted_s2_cells = labels_map[predicted_class_indices]
        predicted_latlons_df = cell_centers.loc[predicted_s2_cells, ['center_latitude', 'center_longitude']]
        predicted_gps.append(torch.tensor(predicted_latlons_df.values, dtype=torch.float32, device=device))
        true_gps.append(gps)
        
        bs = x.size(0)
        total_loss += loss.detach() * bs              # tensor, stays on GPU
        total_acc  += acc.detach() * bs               # make sure acc is a tensor
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
    
    loss_avg = (total_loss / max(n, 1)).item()
    acc_avg  = (total_acc  / max(n, 1)).item()
    
    return {"loss": loss_avg, "acc": acc_avg, "geo_acc": geo_acc_avg}


def _get_pbar(loader, desc: str, enable: bool = True):
    if not enable:
        return loader
    try:
        from tqdm.auto import tqdm
        return tqdm(loader, desc=desc, leave=False)
    except Exception:
        return loader

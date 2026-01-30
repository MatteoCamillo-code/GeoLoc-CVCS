import torch
from torch.cuda.amp import autocast
from metrics.classification import accuracy_top1
from metrics.geospatial import haversine_km, geo_accuracy, get_predicted_gps, get_weighted_predicted_gps
import training.engine_utils as utils

def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    loss_weights, 
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
        x, labels, gps = utils.to_device(batch, device)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
            logits = utils.compute_logits(model, x)
            loss = utils.compute_loss(model, logits, labels, criterion, loss_weights)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        acc = utils.compute_accuracy(logits, labels)

        bs = x.size(0)
        total_loss = utils.update_running_metric(total_loss, loss, bs)  # tensor, stays on GPU
        total_acc = utils.update_running_metric(total_acc, acc, bs)      # make sure acc is a tensor
        n += bs

        if use_tqdm and (step % 40 == 0):
            it.set_postfix(
                loss=f"{loss.detach().float().item():.4f}",
                acc=f"{(acc.detach().float().item()*100):.2f}%",
            )   
    loss_avg, acc_avg = utils.finalize_epoch_metrics(total_loss, total_acc, n)
    return {"loss": loss_avg, "acc": acc_avg}

@torch.inference_mode()
def evaluate(
    model,
    loader,
    cell_centers,
    cells_hierarchy,
    labels_map,
    device: torch.device,
    gps_method: str = "weighted",
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
        x, labels, gps = utils.to_device(batch, device)

        with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
            logits = utils.compute_logits(model, x)

        acc = utils.compute_accuracy(logits, labels)
        
        if gps_method == "weighted":
            # Collect GPS predictions and ground truth (keep on GPU)
            predicted_gps.append(
                get_weighted_predicted_gps(logits, cells_hierarchy, labels_map, 1, device)
            )
        else:
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
                predicted_class_indices = logits[0].argmax(dim=1).cpu().numpy()
            predicted_gps.append(
                get_predicted_gps(predicted_class_indices, cell_centers, labels_map, device)
            )
        
        true_gps.append(gps)
        
        bs = x.size(0)
        total_loss = utils.update_running_metric(total_loss, loss, bs)  # tensor, stays on GPU
        total_acc = utils.update_running_metric(total_acc, acc, bs)      # make sure acc is a tensor
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
    
    loss_avg, acc_avg = utils.finalize_epoch_metrics(total_loss, total_acc, n)
    
    return {"loss": loss_avg, "acc": acc_avg, "geo_acc": geo_acc_avg}


def _get_pbar(loader, desc: str, enable: bool = True):
    if not enable:
        return loader
    try:
        from tqdm.auto import tqdm
        return tqdm(loader, desc=desc, leave=False)
    except Exception:
        return loader

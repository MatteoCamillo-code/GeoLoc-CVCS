import torch

def to_device(batch, device):
    # dataset.osv_dataset.OSV_mini returns: (image, labels, gps)
    # labels shape: [B, 3] for 3 label configs
    x, labels, gps = batch
    return x.to(device, non_blocking=True), labels.to(device, non_blocking=True), gps.to(device, non_blocking=True) if gps is not None else None

def compute_logits(model, x):
    logits = model.get_coarse_level_logits(x)
    return logits

def compute_loss(model, logits, labels, criterion, loss_weights):
    if loss_weights:
      individual_loss = torch.stack([criterion(logit, labels[:, idx])
                            for idx, logit in enumerate(logits)])
      loss =(individual_loss*loss_weights).sum()
    else:
      loss = torch.stack([criterion(logit, labels[:, idx])
                            for idx, logit in enumerate(logits)]).mean()
    return loss

def compute_accuracy(logits, labels):
    return torch.stack([(logit.argmax(1) == labels[:, idx]).float().mean()
                        for idx, logit in enumerate(logits)]).mean()

def update_running_metric(total, value, batch_size):
    total += value.detach() * batch_size
    return total


def finalize_epoch_metrics(total_loss, total_acc, n):
    loss_avg = (total_loss / max(n, 1)).item()
    acc_avg = (total_acc / max(n, 1)).item()
    return loss_avg, acc_avg

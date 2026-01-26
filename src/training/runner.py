from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
from time import time

from training.engine import train_one_epoch, evaluate
from training.callbacks import EarlyStopping
from utils.checkpointing import save_checkpoint
from utils.logging import get_logger
from utils.paths import abs_path

def fit(cfg, model, train_loader, val_loader, optimizer, criterion, scaler, scheduler=None, label_idx: int = 0, use_tqdm: bool = True):
    logger = get_logger(log_file=str(abs_path(cfg.output_dir, "results", "train.log")))

    es = EarlyStopping(patience=cfg.patience)
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    ckpt_path = abs_path(cfg.output_dir, "checkpoints", cfg.model_name)
    
    logger.info("Starting training...")

    for epoch in range(cfg.max_epochs):
        start_time = time()
        
        tr = train_one_epoch(model, train_loader, optimizer, criterion, scaler, cfg.device, amp=cfg.amp, use_tqdm=use_tqdm)
        va = evaluate(model, val_loader, criterion, cfg.device, amp=cfg.amp, use_tqdm=use_tqdm)

        epoch_time = time() - start_time

        if scheduler is not None:
            scheduler.step(va["loss"])

        history["train_loss"].append(tr["loss"])
        history["train_acc"].append(tr["acc"])
        history["val_loss"].append(va["loss"])
        history["val_acc"].append(va["acc"])

        logger.info(
            f"Epoch {epoch+1}/{cfg.max_epochs} | "
            f"train loss={tr['loss']:.4f} acc={(tr['acc']*100):.2f}% | "
            f"val loss={va['loss']:.4f} acc={(va['acc']*100):.2f}% | "
            f"time={epoch_time:.2f}s"
        )

        # save best
        if va["acc"] >= es.best:
            save_checkpoint(
                ckpt_path, model, optimizer, epoch,
                extra={"cfg": asdict(cfg), "best_val_acc": va["acc"]}
            )

        if es.step(va["acc"]):
            logger.info("Early stopping.")
            break

    return history

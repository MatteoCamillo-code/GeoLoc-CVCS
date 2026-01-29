from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
from time import time

from training.engine import train_one_epoch, evaluate
from training.callbacks import EarlyStopping
from utils.checkpointing import save_checkpoint
from utils.logging import get_logger
from utils.paths import abs_path
from utils.io import save_json

def fit(cfg, model, data_loader, cell_centers, cells_hierarchy, 
        optimizer, criterion, scaler, scene, scheduler=None, 
        use_tqdm: bool = True, logger=None, history_path=None, version: int = 0,):
    if logger is None:
        logger = get_logger(log_file=str(abs_path(cfg.output_dir, "logs", "train.log")))

    es = EarlyStopping(patience=cfg.patience, delta_patience=cfg.delta_patience)
    
    val_loader = data_loader["val_loader"]
    train_loader = data_loader["train_loader"]
    labels_map_dict = data_loader["label_maps"]
    
    history = {"train_loss": [], "train_acc": [], "val_loss": [], 
               "val_acc": [], "geo_acc": [], "val_size": len(val_loader.dataset), "train_size": len(train_loader.dataset)}

    ckpt_path = abs_path(cfg.output_dir, "checkpoints", (cfg.model_name + f"_{scene}_v{version}.pt"))
    
    logger.info(f"Starting training {cfg.model_name} for scene {scene} ...")

    for epoch in range(cfg.max_epochs):
        start_time = time()
        
        tr = train_one_epoch(model, train_loader, optimizer, criterion, scaler, cfg.device, 
                             amp=cfg.amp, use_tqdm=use_tqdm)
        va = evaluate(model, val_loader, cell_centers, cells_hierarchy, 
                      labels_map_dict, criterion, cfg.device, gps_method=cfg.gps_method, 
                      amp=cfg.amp, use_tqdm=use_tqdm)

        epoch_time = time() - start_time

        if scheduler is not None:
            scheduler.step()
            
        history["train_loss"].append(tr["loss"])
        history["train_acc"].append(tr["acc"])
        history["val_loss"].append(va["loss"])
        history["val_acc"].append(va["acc"])
        history["geo_acc"].append(va["geo_acc"])

        logger.info(
            f"Epoch {epoch+1}/{cfg.max_epochs} | "
            f"train loss={tr['loss']:.4f} acc={(tr['acc']*100):.2f}% | "
            f"val loss={va['loss']:.4f} acc={(va['acc']*100):.2f}% | "
            f"geo acc={va['geo_acc']} | "
            f"time={epoch_time:.1f}s"
        )

        # save history after each epoch
        if history_path is not None:
            save_json(obj=history, path=history_path)

        # save best
        if va["acc"] >= es.best:
            save_checkpoint(
                ckpt_path, model, optimizer, epoch,
                extra={"cfg": asdict(cfg), "best_val_acc": va["acc"]}
            )

        if es.step(va["acc"]):
            logger.info("Early stopping.")
            break
    
    logger.info("Training completed.")
    return history

from dataclasses import dataclass, field

@dataclass
class TrainConfig:
    seed: int = 42
    batch_size: int = 128
    num_workers: int = 4
    prefetch_factor: int = 2
    lr: float = 2e-3
    momentum: float = 0.9
    weight_decay: float = 1e-4
    scheduler_step_size: int = 3
    scheduler_gamma: float = 0.1
    max_epochs: int = 30
    patience: int = 6
    delta_patience: float = 1e-3
    amp: bool = True
    device: str = "cuda"
    model_name: str = "MH_inceptionv4_wd_sp_30"
    use_tqdm: bool = True
    dropout: float = 0.2
    use_cbam: bool = False
    cbam_reduction: int = 16
    
    label_smoothing: float = 0.2
    
    gps_method: str = "weighted"  # "weighted", "argmax"
    
    backbone: str = "inceptionv4"  # backbone model name
    image_size: int = 299  # input image size for the model
    
    scenes: list[str] = field(default_factory=lambda: ["total"])  # scenes to be used for ISN classification
    same_partitions: bool = True  # whether to use same partitions for all coarse labels
    coarse_label_idx: list[int] = field(default_factory=lambda: [0, 1, 2])  # indices of the labels to be used for multi-head classification
    
    train_size_pct: float = 100.0 
    val_size_pct: float = 100.0
    

    # IMPORTANT: relative-to-root output folder name
    output_dir: str = "outputs"

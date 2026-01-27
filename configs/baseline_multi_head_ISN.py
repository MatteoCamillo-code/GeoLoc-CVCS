from dataclasses import dataclass, field

@dataclass
class TrainConfig:
    seed: int = 42
    batch_size: int = 128
    num_workers: int = 6
    lr: float = 5e-4
    momentum: float = 0.9
    weight_decay: float = 1e-4
    scheduler_step_size: int = 5
    scheduler_gamma: float = 0.5
    max_epochs: int = 1
    patience: int = 5
    delta_patience: float = 1e-3
    amp: bool = True
    device: str = "cuda"
    model_name: str = "baseline_multi_head_ISN"
    use_tqdm: bool = True
    scenes: list[str] = field(default_factory=lambda: ["urban", "natural"])  # scenes to be used for ISN classification
    
    dropout: float = 0.0 
    
    coarse_label_idx: list[int] = field(default_factory=lambda: [0, 1, 2])  # indices of the labels to be used for multi-head classification

    # IMPORTANT: relative-to-root output folder name
    output_dir: str = "outputs"

# GeoLoc-CVCS

A deep learning-based geolocation system that predicts GPS coordinates from images using hierarchical multi-head classification with S2 geometry partitioning.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Configuration Files](#configuration-files)
- [Using the Notebooks](#using-the-notebooks)
- [Training Pipeline](#training-pipeline)
- [Data Structure](#data-structure)
- [Outputs](#outputs)

## Overview

This project implements a Computer Vision-based Coordinate System (CVCS) for geolocation prediction. It uses:

- **Multi-head classifiers** with hierarchical S2 cell partitioning
- **Multiple backbones**: ResNet50 and InceptionV4
- **CBAM attention mechanisms** for improved feature extraction
- **Scene-specific models**: Support for training on different scene types (natural, urban, total)
- **Weighted loss** and label smoothing for handling class imbalance

## Project Structure

```
GeoLoc-CVCS/
│
├── configs/                    # Training configuration files
│   ├── baseline.py             # Single-head baseline configuration
│   ├── baseline_multi_head.py  # Multi-head ResNet50 configuration
│   ├── baseline_multi_head_inceptionv4.py  # Multi-head InceptionV4 config
│   ├── baseline_multi_head_ISN.py          # ISN (Image Scene Network) config
│   └── baseline_multi_head_ISN_inceptionv4.py
│
├── data/                       # Dataset storage
│   ├── classify/               # Classification test data
│   ├── metadata/               # Processed metadata files
│   │   ├── original-datasets/  # Original dataset metadata
│   │   ├── places-classification/  # Scene classification results
│   │   └── s2-geo-cells/       # S2 geometry cell partitions
│   ├── raw/                    # Raw image data
│   │   ├── mp16_images/        # MP-16 dataset images
│   │   └── osv5m/              # OpenStreetView 5M dataset
│   └── splits/                 # Train/validation splits
│
├── outputs/                    # Training outputs and results
│   ├── checkpoints/            # Saved model weights (.pt files)
│   ├── graphs/                 # Training visualization graphs
│   ├── history/                # Training history JSON files
│   ├── logs/                   # Training logs
│   └── gps_predictions*.csv    # GPS prediction results
│
├── src/                        # Source code
│   ├── dataset/                # Dataset and dataloader utilities
│   ├── map_partitioning/       # S2 geometry partitioning
│   │   └── s2/                 # S2 cell utilities
│   ├── metrics/                # Evaluation metrics
│   ├── models/                 # Model architectures
│   │   ├── cbam.py             # CBAM attention module
│   │   ├── classifier.py       # Single-head classifier
│   │   ├── multi_head_classifier.py     # Multi-head classifier
│   │   └── multi_head_classifier_cbam.py # Multi-head with CBAM
│   ├── notebooks/              # Jupyter notebooks for training/testing
│   ├── pipelines/              # Inference and evaluation pipelines
│   ├── scene_classification/   # Scene type classification
│   ├── torch_modules/          # PyTorch modules
│   ├── training/               # Training engine and utilities
│   └── utils/                  # General utilities
│
├── requirements.txt            # Python dependencies
├── scene_hierarchy_places365.csv  # Places365 scene hierarchy
└── README.md                   # This file
```

### Main Folders Description

#### `configs/`
Contains Python dataclass-based configuration files that define all hyperparameters and training settings. Each config file represents a different model architecture or training strategy.

#### `data/`
- **`raw/`**: Store raw image datasets here (MP-16, OSV5M)
- **`metadata/`**: Processed CSV files with image metadata, scene classifications, and S2 cell assignments
- **`splits/`**: Train/validation split definitions
- **`classify/`**: Test data for classification tasks

#### `src/`
Core source code organized by functionality:
- **`models/`**: Neural network architectures
- **`dataset/`**: Data loading and augmentation
- **`training/`**: Training loops, losses, and callbacks
- **`pipelines/`**: End-to-end inference pipelines
- **`metrics/`**: Evaluation metrics (geospatial distance, classification accuracy)
- **`map_partitioning/`**: S2 geometry-based spatial partitioning

#### `outputs/`
All training artifacts:
- **`checkpoints/`**: Saved model weights (`.pt` files)
- **`history/`**: Training metrics per epoch (JSON format)
- **`logs/`**: Detailed training logs
- **`graphs/`**: Training curves and visualizations

## Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)
- 16GB+ RAM

### Setup

1. Clone the repository:
```bash
git clone https://github.com/MatteoCamillo-code/GeoLoc-CVCS.git
cd GeoLoc-CVCS
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Configuration Files

Configuration files in the `configs/` folder use Python dataclasses to define all training parameters. Each config file contains a `TrainConfig` class with the following key parameters:

### Key Configuration Parameters

```python

    seed                # Random seed for reproducibility
    batch_size          # Batch size for training
    num_workers         # Number of dataloader workers
    lr                  # Learning rate
    weight_decay        # L2 regularization
    max_epochs          # Maximum training epochs
    patience            # Early stopping patience
    
    # Model architecture
    backbone            # "resnet50" or "inceptionv4"
    image_size          # Input image size (224 for ResNet, 299 for Inception)
    use_cbam            # Enable CBAM attention
    dropout             # Dropout rate
    
    # Multi-head classification
    coarse_label_idx    # Hierarchical levels to use
    scenes              # Scene types: "natural", "urban", "total"
    same_partitions     # Use same S2 partitions for all scenes
    
    # Loss configuration
    label_smoothing     # Label smoothing factor
    weighted_loss       # Use class-weighted loss
    
    # Data
    expanded_dataset    # Use expanded dataset
    train_size_pct      # Percentage of training data to use
    val_size_pct        # Percentage of validation data to use
    
    # Output
    model_name          # Model name for saving
    output_dir          # Output directory
```

### Available Configurations

1. **`baseline.py`**: Single-head baseline model
2. **`baseline_multi_head.py`**: Multi-head ResNet50 with CBAM
3. **`baseline_multi_head_inceptionv4.py`**: Multi-head InceptionV4 with CBAM
4. **`baseline_multi_head_ISN.py`**: Image Scene Network with ResNet50
5. **`baseline_multi_head_ISN_inceptionv4.py`**: ISN with InceptionV4

### How to Use Configs

Import the desired configuration in the training notebook:

```python
from configs.desired_config_file import TrainConfig
cfg = TrainConfig()
```

You can also create custom configurations by copying and modifying existing config files.

## Using the Notebooks

Notebooks are located in [`src/notebooks/`](src/notebooks/). The main training notebook is [`Baseline_multihead.ipynb`](src/notebooks/Baseline_multihead.ipynb).

### Baseline_multihead.ipynb

This notebook provides a complete training pipeline for multi-head geolocation models.

#### Notebook Structure

1. **Path Setup**: Configures paths and multiprocessing (required for Windows)
2. **Imports**: Loads all necessary libraries and the configuration
3. **Data Loading**: Loads S2 cell metadata and creates dataloaders
4. **Model Initialization**: Creates multi-head classifier models for each scene
5. **Training Loop**: Trains models with early stopping and checkpointing
6. **Checkpoint Consolidation**: Saves all scene models into a single checkpoint file

#### How to Use

1. **Configure file paths** (Second Cell):
```python
# FOR LOCAL USE THIS LINES
current = Path.cwd()
src_path = current / "src" if (current / "src").exists() else current.parent
```
```python
# FOR COLAB USE THIS LINE INSTEAD
BRANCH_NAME = "branch_name"  # Change this to switch branches
!git clone -b {BRANCH_NAME} https://github.com/MatteoCamillo-code/GeoLoc-CVCS.git
!cd /content/GeoLoc-CVCS && git pull origin {BRANCH_NAME} && cd ..
src_path = Path("/content/GeoLoc-CVCS/src").resolve()
```

2. **Select Configuration**:
```python
from configs.desired_config_file import TrainConfig
```

3. **Run All Cells**:
   - The notebook will automatically:
     - Load and prepare data
     - Create dataloaders with proper augmentation
     - Initialize models for each scene
     - Train with early stopping
     - Save checkpoints and training history

4. **Monitor Training**:
   - Progress bars show epoch-level metrics
   - Logs are saved to [`outputs/logs/train.log`](outputs/logs/train.log)
   - Training history is saved as JSON after each epoch

#### Scene-Specific Training

The notebook supports training separate models for different image scene types (it can be set in the config file):

```python
# Train on all scenes combined
scenes: list[str] = field(default_factory=lambda: ["total"])

# Train separate models for natural and urban scenes
scenes: list[str] = field(default_factory=lambda: ["natural", "urban"])

# Train on a specific scene only
scenes: list[str] = field(default_factory=lambda: ["urban"])
```

#### Output Files

After training, the notebook generates:

- **Checkpoint**: [`outputs/checkpoints/{model_name}_v{version}.pt`](outputs/checkpoints/)
- **History**: [`outputs/history/{model_name}_v{version}.json`](outputs/history/)
- **Logs**: [`outputs/logs/train.log`](outputs/logs/train.log)

The checkpoint contains:
```python
{
    "scene1": {
        "model_state": ...,
        "optimizer_state": ...,
        "label_maps": ...,
        "epoch": ...,
        "cfg": ..., # configuration used
        "val_loss": ...
    },
    "scene2": {...},
    ...
}
```

### Other Notebooks

- **`pipeline.ipynb`**: End-to-end inference pipeline for geolocation prediction
- **`model_loader.ipynb`**: Load and inspect trained models
- **`s2_configuration.ipynb`**: Configure S2 cell partitioning
- **`download_and_classify_with_threshold.ipynb`**: Image download and classification

## Training Pipeline

The training pipeline ([`src/training/runner.py`](src/training/runner.py)) includes:

1. **Data Loading**: Multi-scene dataloaders with augmentation
2. **Model Training**: 
   - Mixed precision training (AMP)
   - Multi-head hierarchical classification
   - Weighted cross-entropy loss with label smoothing
3. **Validation**: Geospatial distance metrics (GCD, GPS2Meter)
4. **Early Stopping**: Based on validation loss
5. **Checkpointing**: Saves best model per scene
6. **Logging**: Comprehensive training logs

### Key Features

- **Hierarchical Classification**: Uses S2 cells at multiple resolution levels
- **CBAM Attention**: Channel and spatial attention mechanisms
- **Scene-Specific Models**: Separate models for different geographic contexts
- **Weighted Loss**: Handles class imbalance in geographic distribution
- **Label Smoothing**: Reduces overfitting on hierarchical labels

## Data Structure

### Required Data Files

The training pipeline expects the following data structure:

1. **Images**: Store in [`data/raw/mp16_images/`](data/raw/mp16_images/) or [`data/raw/osv5m/`](data/raw/osv5m/)

2. **Metadata CSVs** in [`data/metadata/s2-geo-cells/`](data/metadata/s2-geo-cells/):
   - `train_val_split_geocells_total_expanded.csv`: Train/val split with corresponding S2 cells
   - `cell_center_dataset_total_expanded.csv`: S2 cell center coordinates
   - `cell_hierarchy_dataset_total_expanded.csv`: Hierarchical S2 cell mappings (used for weighted gps estimation)

## Outputs

### Checkpoints

Model checkpoints are saved to [`outputs/checkpoints/`](outputs/checkpoints/):
- Format: `{model_name}_v{version}.pt`
- Contains: model state, optimizer state, training metrics

### Training History

JSON files in [`outputs/history/`](outputs/history/) contain:
- Epoch-wise training and validation metrics
- Accuracy in terms of geospatial distance thresholds

### GPS Predictions

CSV files with predicted coordinates from pipeline.ipynb:
- `gps_predictions.csv`: General predictions


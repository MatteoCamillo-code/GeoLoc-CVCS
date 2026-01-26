"""
Utility functions for creating DataLoaders with optional dataset subsetting.
"""

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from dataset.osv_dataset import OSV_mini, seed_worker, fast_collate


def create_transforms(img_size=224, augment=True):
    """Create train and validation transforms."""
    if augment:
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0), ratio=(3/4, 4/3)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ColorJitter(0.1, 0.1, 0.1, 0.05),
            transforms.RandomApply([transforms.RandomRotation(10)], p=0.2),
            transforms.ToTensor(),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

    return train_transform, val_transform


def create_dataloaders(
    image_root,
    csv_path,
    batch_size=128,
    num_workers=0,
    img_size=224,
    seed=42,
    train_subset_pct=100.0,
    val_subset_pct=100.0,
    augment=True,
    prefetch_factor=4,
    persistent_workers=True,
):
    """
    Create train and validation DataLoaders with optional dataset subsetting.

    Parameters
    ----------
    image_root : str
        Root directory containing images organized by region.
    csv_path : str or Path
        Path to CSV with split and metadata.
    batch_size : int, default=128
        Batch size for DataLoaders.
    num_workers : int, default=0
        Number of workers for multiprocessing (0 = no multiprocessing).
    img_size : int, default=224
        Image size for resizing.
    seed : int, default=42
        Random seed for reproducibility.
    train_subset_pct : float, default=100.0
        Percentage of training set to use (0-100).
    val_subset_pct : float, default=100.0
        Percentage of validation set to use (0-100).
    augment : bool, default=True
        Whether to apply data augmentation to training set.
    prefetch_factor : int, default=4
        Number of batches to prefetch per worker.
    persistent_workers : bool, default=True
        Whether to keep workers alive between epochs.

    Returns
    -------
    dict
        Dictionary with keys:
        - 'train_loader': DataLoader for training
        - 'val_loader': DataLoader for validation
        - 'label_maps': Dictionary mapping label columns to categories
        - 'train_size': Number of training samples used
        - 'val_size': Number of validation samples used
    """
    # Create transforms
    train_transform, val_transform = create_transforms(img_size, augment=augment)

    # Create full datasets
    train_dataset = OSV_mini(
        image_root=image_root,
        csv_path=csv_path,
        transform=train_transform,
        split="train",
        scene="total",
        label_maps=None,
    )

    val_dataset = OSV_mini(
        image_root=image_root,
        csv_path=csv_path,
        transform=val_transform,
        split="val",
        scene="total",
        label_maps=train_dataset.label_maps,
    )

    # Apply subsetting if requested
    if train_subset_pct < 100.0:
        train_size = int(len(train_dataset) * train_subset_pct / 100.0)
        indices = torch.randperm(len(train_dataset))[:train_size].tolist()
        train_dataset = Subset(train_dataset, indices)

    if val_subset_pct < 100.0:
        val_size = int(len(val_dataset) * val_subset_pct / 100.0)
        indices = torch.randperm(len(val_dataset))[:val_size].tolist()
        val_dataset = Subset(val_dataset, indices)

    # Create generator for reproducibility
    g = torch.Generator()
    g.manual_seed(seed)

    # Adjust worker settings
    use_workers = num_workers > 0
    prefetch = prefetch_factor if use_workers else None
    persistent = persistent_workers if use_workers else False

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=prefetch,
        persistent_workers=persistent,
        collate_fn=fast_collate,
        worker_init_fn=seed_worker if use_workers else None,
        generator=g,
    )

    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=prefetch,
        persistent_workers=persistent,
        collate_fn=fast_collate,
        worker_init_fn=seed_worker if use_workers else None,
    )

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "label_maps": train_dataset.dataset.label_maps if isinstance(train_dataset, Subset) else train_dataset.label_maps,
        "train_size": len(train_dataset),
        "val_size": len(val_dataset),
    }

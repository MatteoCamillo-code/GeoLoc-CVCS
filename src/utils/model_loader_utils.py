import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import timm

from src.models.multi_head_classifier import MultiHeadClassifier
from src.utils.paths import get_current_version
from src.metrics.geospatial import get_predicted_gps, get_weighted_predicted_gps, haversine_km
from torchvision.models import resnet50, ResNet50_Weights



def get_val_samples(loader_dict, scenes, n_samples=10):
    """
    Extract n validation samples from each scene.
    
    Args:
        loader_dict: Dictionary of dataloaders for each scene
        scenes: List of scene names
        n_samples: Number of samples to extract per scene
    
    Returns:
        Dictionary with scene names as keys, containing lists of (images, labels, gps) tuples
    """
    samples_by_scene = {}
    
    for scene in scenes:
        samples = []
        val_loader = loader_dict[scene]["val_loader"]
        
        for batch in val_loader:
            images, labels, gps = batch
            
            # Add individual samples from the batch
            for i in range(images.shape[0]):
                samples.append({
                    'image': images[i],
                    'labels': labels[i],
                    'gps': gps[i],
                })
                
                if len(samples) >= n_samples:
                    break
            
            if len(samples) >= n_samples:
                break
        
        samples_by_scene[scene] = samples[:n_samples]
    
    return samples_by_scene

def print_samples(val_samples):
    # Print sample information
    text = ""

    for scene, samples in val_samples.items():
        text += f"\n{'='*60}"
        text += f"\nScene: {scene}"
        text += f"\n{'='*60}"
        text += f"\nNumber of samples: {len(samples)}"
        
        if samples:
            # Print first sample details
            sample = samples[0]
            text += f"\nFirst sample details:"
            text += f"\n  Image shape: {sample['image'].shape}"
            text += f"\n  Image dtype: {sample['image'].dtype}"
            text += f"\n  Labels shape: {sample['labels'].shape}"
            text += f"\n  Labels: {sample['labels'].tolist()}"
            text += f"\n  GPS shape: {sample['gps'].shape}"
            text += f"\n  GPS (lat, lon): {sample['gps'].tolist()}"
    # Display first image from each scene
    text += f"\n{'='*60}"
    text += "Displaying first image from each scene:"
    text += f"\n{'='*60}"

    num_scenes = len(val_samples)
    fig, axes = plt.subplots(1, num_scenes, figsize=(5*num_scenes, 5))

    if num_scenes == 1:
        axes = [axes]

    for idx, (scene, samples) in enumerate(val_samples.items()):
        if samples:
            img = samples[0]['image']
            # Denormalize if needed (assuming images are in [0, 1] range)
            if img.max() <= 1.0:
                img = img
            else:
                img = img / 255.0
            
            # Convert from CHW to HWC for display
            img_display = img.permute(1, 2, 0).cpu().numpy()
            
            axes[idx].imshow(img_display)
            axes[idx].set_title(f"{scene}\nLabels: {samples[0]['labels'].tolist()}")
            axes[idx].axis('off')

    plt.tight_layout()
    return plt, text


# LOAD MODELS FROM CONSOLIDATED CHECKPOINT

from src.utils.checkpointing import load_checkpoint

def create_backbone(backbone_type):
    """
    Create and return a backbone model.
    
    Args:
        backbone_type: Type of backbone ('inceptionv4' or 'resnet50')
    
    Returns:
        tuple: (backbone model, feature dimension)
    """
    if backbone_type == "inceptionv4":
        backbone = timm.create_model('inception_v4', pretrained=True)
        feat_dim = 1536
    elif backbone_type == "resnet50":
        weights = ResNet50_Weights.IMAGENET1K_V2
        backbone = resnet50(weights=weights)
        feat_dim = 2048
    else:
        raise ValueError(f"Unsupported backbone type: {backbone_type}")
    
    # Remove final classification layer and flatten
    backbone = nn.Sequential(*list(backbone.children())[:-1], nn.Flatten(1))
    
    return backbone, feat_dim


def create_model_for_scene(scene, loader_dict, cfg, device):
    """
    Create a MultiHeadClassifier model for a specific scene.
    
    Args:
        scene: Scene name
        loader_dict: Dictionary containing dataloaders and label maps
        cfg: Configuration object
        device: Device to place model on
    
    Returns:
        MultiHeadClassifier: The initialized model
    """
    backbone, feat_dim = create_backbone(cfg.backbone)
    backbone = backbone.to(device)
    
    # Get number of classes for each head
    num_classes = list(map(
        lambda idx: len(loader_dict[scene]["label_maps"][f"label_config_{idx + 1}"]),
        cfg.coarse_label_idx
    ))
    
    model = MultiHeadClassifier(
        backbone=backbone,
        feat_dim=feat_dim,
        head_dims=num_classes,
        dropout=cfg.dropout,
        coarse_level_idx=cfg.coarse_label_idx,
    ).to(device)
    
    return model


def load_models_from_consolidated_checkpoint(checkpoint_dir, cfg, loader_dict, device):
    """
    Load all scene models from a consolidated checkpoint file.
    
    Args:
        checkpoint_dir: Path to checkpoint directory
        cfg: Configuration object
        loader_dict: Dictionary containing dataloaders and label maps
        device: Device to load models on
    
    Returns:
        dict: Dictionary mapping scene names to loaded models
    """
    # Get the consolidated checkpoint version
    base_name = f"{cfg.model_name}"
    version = get_current_version(checkpoint_dir, base_name, extension="pt")
    ckpt_path = checkpoint_dir / f"{base_name}_v{version}.pt"
    
    print(f"Loading consolidated checkpoint: {ckpt_path}")
    
    # Load consolidated checkpoint containing all scenes
    consolidated_ckpt = torch.load(ckpt_path, map_location=device)
    
    # Create and load models for each scene
    models = {}
    for scene in cfg.scenes:
        print(f"\n--- Loading model for scene: {scene} ---")
        
        # Create model architecture
        model = create_model_for_scene(scene, loader_dict, cfg, device)
        
        # Load checkpoint for this scene
        if scene in consolidated_ckpt:
            model.load_state_dict(consolidated_ckpt[scene]["model"])
            model.eval()
            print(f"Loaded epoch: {consolidated_ckpt[scene].get('epoch')}")
        else:
            print(f"Warning: No checkpoint found for scene {scene}")
        
        models[scene] = model
    
    print(f"\nSuccessfully loaded {len(models)} models for scenes: {list(models.keys())}")
    
    return models

# Compute predictions for validation samples

def predict_samples(models, val_samples, loader_dict, cell_centers_dfs, cells_hierarchy_dfs, cfg, device, gps_method="weighted", top_k=1):
    """
    Generate predictions for validation samples using the loaded models.
    
    Args:
        models: Dictionary of models for each scene
        val_samples: Dictionary of validation samples for each scene
        loader_dict: Dictionary containing label maps
        cell_centers_dfs: Dictionary of cell centers DataFrames for each scene
        cells_hierarchy_dfs: Dictionary of cells hierarchy DataFrames for each scene
        cfg: Configuration object
        device: Device to run inference on
        gps_method: Method for GPS prediction ('weighted' or 'argmax')
        top_k: Number of top paths to use for weighted GPS prediction
    
    Returns:
        Dictionary with predictions for each scene
    """
    predictions = {}
    
    for scene in val_samples.keys():
        model = models[scene]
        samples = val_samples[scene]
        label_maps = loader_dict[scene]["label_maps"]
        cells_hierarchy = cells_hierarchy_dfs[scene]
        cell_centers = cell_centers_dfs[scene]
        
        scene_predictions = []
        
        with torch.no_grad():
            for i, sample in enumerate(samples):
                # Prepare input
                image = sample['image'].unsqueeze(0).to(device)  # Add batch dimension
                
                # Get prediction
                outputs = model(image)  # Returns list of logits for each head
                
                # Process each head's prediction
                predicted_labels = []
                confidences = []
                label_names = []
                
                for head_idx, logits in enumerate(outputs):
                    probs = torch.softmax(logits, dim=1)
                    confidence, pred_idx = torch.max(probs, dim=1)
                    
                    predicted_labels.append(pred_idx.item())
                    confidences.append(confidence.item())
                    
                    # Get actual label name from label map
                    label_config_name = f"label_config_{cfg.coarse_label_idx[head_idx] + 1}"
                    label_map = label_maps[label_config_name]
                    label_name = label_map[pred_idx.item()]
                    label_names.append(label_name)
                
                # Get ground truth labels
                true_labels = sample['labels'].tolist()
                true_label_names = []
                for head_idx, true_idx in enumerate(true_labels):
                    label_config_name = f"label_config_{cfg.coarse_label_idx[head_idx] + 1}"
                    label_map = label_maps[label_config_name]
                    if true_idx >= 0:  # Check if not a missing label (-1)
                        true_label_names.append(label_map[true_idx])
                    else:
                        true_label_names.append("Unknown")
                
                # Predict GPS coordinates using hierarchical method
                if gps_method == "weighted":
                    predicted_gps = get_weighted_predicted_gps(
                        outputs, cells_hierarchy, label_maps, top_k, device
                    )
                else:
                    # Use argmax method (first head only)
                    predicted_class_indices = outputs[0].argmax(dim=1).cpu().numpy()
                    predicted_s2_cells = label_maps.get("label_config_1")[predicted_class_indices]
                    predicted_latlons_df = cell_centers.loc[predicted_s2_cells, ['center_latitude', 'center_longitude']]
                    predicted_gps = torch.tensor(predicted_latlons_df.values, dtype=torch.float32, device=device)
                
                # Get true GPS
                true_gps = sample['gps'].unsqueeze(0).to(device)
                
                # Calculate distance error
                distance_km = haversine_km(predicted_gps, true_gps)
                
                # Store prediction result
                result = {
                    'sample_idx': i,
                    'image': sample['image'],
                    'true_gps': sample['gps'].tolist(),
                    'predicted_gps': predicted_gps.squeeze(0).cpu().tolist(),
                    'distance_km': distance_km.item(),
                    'true_labels': true_labels,
                    'true_label_names': true_label_names,
                    'predicted_labels': predicted_labels,
                    'predicted_label_names': label_names,
                    'confidences': confidences,
                }
                
                scene_predictions.append(result)
        
        predictions[scene] = scene_predictions
    
    return predictions

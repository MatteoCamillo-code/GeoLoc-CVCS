import os
import csv
import numpy as np
import torch
import torchvision
import torchvision.transforms as tfm
import torch.nn.functional as F
from PIL import Image

# Relative import to reach downloader.py in the parent 'src' folder
from ..downloader import download_scene_hierarchy_file, download_pretrained_on_places

class SceneClassifier(torch.nn.Module):
    def __init__(self, scene_hierarchy_file='scene_hierarchy_places365.csv', model_name="resnet50"):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Ensure resources are available
        if not os.path.exists(scene_hierarchy_file): 
            download_scene_hierarchy_file()
        if not os.path.exists(f"{model_name}_places365.pth.tar"): 
            download_pretrained_on_places(model_name)
        
        # Load and normalize hierarchy metadata
        hierarchy = []
        with open(scene_hierarchy_file, 'r', encoding='utf-8') as f:
            content = csv.reader(f)
            next(content)  # Skip header
            next(content)  # Skip second header line
            for line in content: 
                hierarchy.append(line[1:4])
        
        hierarchy = np.asarray(hierarchy, dtype=float)
        # Normalize rows to sum to 1
        self.hierarchy_places3 = hierarchy / np.expand_dims(np.sum(hierarchy, axis=1).clip(min=1.0), axis=-1)
        
        # Load Model Architecture
        self.model = torchvision.models.resnet50(weights=None)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 365)
        
        # Load Pretrained Weights
        checkpoint = torch.load(f"{model_name}_places365.pth.tar", map_location='cpu', weights_only=False)
        state_dict = {k.replace("module.", ""): v for k, v in checkpoint["state_dict"].items()}
        self.model.load_state_dict(state_dict)
        self.model.to(self.device).eval()

        # Image Preprocessing Transform
        self.transform = tfm.Compose([
            tfm.Resize((256, 256)),
            tfm.CenterCrop(224),
            tfm.ToTensor(),
            tfm.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def classify_one_image(self, image_path):
        """Classifies a single image path and returns the string label."""
        if not os.path.exists(image_path):
            return "File not found"
        img = Image.open(image_path).convert('RGB')
        tensor = self.transform(img).unsqueeze(0).to(self.device)
        # Returns a list of one index, so we take [0]
        prediction_idx = self.forward(tensor)[0]
        return self.label_int_to_str(prediction_idx)

    def batch_classify(self, dataloader):
        """Processes a DataLoader and returns a list of result dictionaries."""
        results = []
        from tqdm import tqdm
        with torch.inference_mode():
            for imgs, paths_list in tqdm(dataloader, desc="Classifying Batches"):
                if imgs.nelement() == 0: 
                    continue
                
                # Use internal forward pass
                preds = self.forward(imgs.to(self.device))
                
                for path, p_idx in zip(paths_list, preds):
                    results.append({
                        'filename': os.path.splitext(os.path.basename(path))[0],
                        'predicted_label': self.label_int_to_str(p_idx)
                    })
        return results

    def forward(self, batch):
        """Internal logic to project 365 scene classes into 3 categories (Indoor, Natural, Urban)."""
        with torch.inference_mode():
            logits = self.model(batch)
            probs = F.softmax(logits, dim=1).cpu().numpy()
            # Dot product with hierarchy matrix
            category_probs = np.matmul(probs, self.hierarchy_places3)
            return np.argmax(category_probs, axis=1).tolist()

    def label_int_to_str(self, idx):
        """Maps integer index back to string labels."""
        mapping = {0: 'Indoor', 1: 'Natural', 2: 'Urban'}
        return mapping.get(idx, 'Unknown')
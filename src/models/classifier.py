import requests
import torch
import torch.nn.functional as F
import numpy as np
import os
import csv
import torchvision
import torchvision.transforms as tfm
from tqdm import tqdm


class SceneClassifierWithConfidence(torch.nn.Module):
    """Scene classifier that returns predictions WITH confidence scores."""
    
    def __init__(self, scene_hierarchy_file='scene_hierarchy_places365.csv', model_name="resnet50"):
        super().__init__()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using Device: {self.device}")
        
        # Download required files
        if not os.path.exists(scene_hierarchy_file):
            download_scene_hierarchy_file()
        if not os.path.exists(f"{model_name}_places365.pth.tar"):
            download_pretrained_on_places(model_name)
        
        # Load hierarchy
        print('Loading scene hierarchy...')
        hierarchy_places3 = []
        with open(scene_hierarchy_file, 'r', encoding='utf-8') as csvfile:
            content = csv.reader(csvfile, delimiter=',')
            next(content); next(content)  # Skip header rows
            for line in content:
                # Handle rows with missing values - only process if we have at least 4 columns
                if len(line) >= 4:
                    try:
                        # Extract columns 1-3 (indices 1, 2, 3) and convert to float
                        values = [float(line[i]) if line[i].strip() else 0.0 for i in range(1, 4)]
                        hierarchy_places3.append(values)
                    except (ValueError, IndexError):
                        # Skip rows that can't be converted to float
                        continue
        
        hierarchy_places3 = np.asarray(hierarchy_places3, dtype=float)
        sum_rows = np.sum(hierarchy_places3, axis=1)
        sum_rows[sum_rows == 0] = 1.0
        self.hierarchy_places3 = hierarchy_places3 / np.expand_dims(sum_rows, axis=-1)
        
        # Load model
        print(f'Loading {model_name}...')
        self.model = torchvision.models.resnet50(weights=None)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 365)
        
        checkpoint = torch.load(f"{model_name}_places365.pth.tar", map_location='cpu')
        state_dict = {k.replace("module.", ""): v for k, v in checkpoint["state_dict"].items()}
        self.model.load_state_dict(state_dict)
        
        self.model.to(self.device)
        self.model.eval()
        
        # Transform
        self.transform = tfm.Compose([
            tfm.Resize((256, 256)),
            tfm.CenterCrop(224),
            tfm.ToTensor(),
            tfm.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    
    def process_images(self, pil_images):
        """Process PIL images and return predictions with confidence."""
        batch = [self.transform(img) for img in pil_images]
        batch = torch.stack(batch).to(self.device)
        return self(batch)
    
    def forward(self, batch):
        """Returns (labels, confidences) instead of just labels."""
        with torch.inference_mode():
            logits = self.model(batch)
            scene_probs = F.softmax(logits, dim=1).cpu().numpy()
            places_probs = np.matmul(scene_probs, self.hierarchy_places3)
            
            # Get both the predicted label AND its confidence
            scene_label_int = np.argmax(places_probs, axis=1)
            confidence = np.max(places_probs, axis=1)  # Max probability = confidence
        
        return scene_label_int.tolist(), confidence.tolist()
    
    def label_int_to_str(self, scene_label_int):
        """Converts integer scene labels to strings."""
        if scene_label_int == 0: return 'Indoor'
        elif scene_label_int == 1: return 'Natural'
        elif scene_label_int == 2: return 'Urban'
        return 'Unknown'
    
def download_file_robust(url, filename):
    """Downloads a file in chunks to prevent corruption/timeouts."""
    if os.path.exists(filename):
        if "pth.tar" in filename and os.path.getsize(filename) < 90 * 1024 * 1024:
            print(f"File {filename} looks corrupt (too small). Re-downloading...")
            os.remove(filename)
        else:
            return

    print(f"Downloading {filename}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))

        with open(filename, "wb") as file, tqdm(
            desc=filename,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(chunk_size=1024):
                size = file.write(data)
                bar.update(size)
        print("Download complete.")
    except Exception as e:
        print(f"Download failed: {e}")
        if os.path.exists(filename):
            os.remove(filename)
        raise
    
def download_scene_hierarchy_file():
    """Downloads the scene hierarchy CSV file."""
    url = 'https://docs.google.com/spreadsheet/ccc?key=1H7ADoEIGgbF_eXh9kcJjCs5j_r3VJwke4nebhkdzksg&output=csv'
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        with open("scene_hierarchy_places365.csv", "w", encoding='utf-8') as file:
            file.write(response.content.decode('utf-8') + "\n")
    except Exception as e:
        print(f"Error downloading hierarchy CSV: {e}")
        raise

def download_pretrained_on_places(model_name="resnet50"):
    """Downloads a pre-trained Places365 model."""
    url = f'http://places2.csail.mit.edu/models_places365/{model_name}_places365.pth.tar'
    download_file_robust(url, f"{model_name}_places365.pth.tar")
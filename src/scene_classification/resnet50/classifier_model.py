import os
import csv
import numpy as np
import torch
import torchvision
import torchvision.transforms as tfm
import torch.nn.functional as F
from PIL import Image
from ..downloader import download_scene_hierarchy_file, download_pretrained_on_places

class SceneClassifier(torch.nn.Module):
    def __init__(self, scene_hierarchy_file='scene_hierarchy_places365.csv', model_name="resnet50"):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if not os.path.exists(scene_hierarchy_file): download_scene_hierarchy_file()
        if not os.path.exists(f"{model_name}_places365.pth.tar"): download_pretrained_on_places(model_name)
        
        hierarchy = []
        with open(scene_hierarchy_file, 'r', encoding='utf-8') as f:
            content = csv.reader(f); next(content); next(content)
            for line in content: hierarchy.append(line[1:4])
        
        hierarchy = np.asarray(hierarchy, dtype=float)
        self.hierarchy_places3 = hierarchy / np.expand_dims(np.sum(hierarchy, axis=1).clip(min=1.0), axis=-1)
        
        self.model = torchvision.models.resnet50(weights=None)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 365)
        checkpoint = torch.load(f"{model_name}_places365.pth.tar", map_location='cpu', weights_only=False)
        self.model.load_state_dict({k.replace("module.", ""): v for k, v in checkpoint["state_dict"].items()})
        self.model.to(self.device).eval()

        self.transform = tfm.Compose([
            tfm.Resize((256, 256)),
            tfm.CenterCrop(224),
            tfm.ToTensor(),
            tfm.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def classify_one_image(self, image_path):
        """Classify a single image path and return string label."""
        img = Image.open(image_path).convert('RGB')
        tensor = self.transform(img).unsqueeze(0).to(self.device)
        return self.label_int_to_str(self.forward(tensor)[0])

    def batch_classify(self, dataloader):
        """The bulk classification loop logic."""
        results = []
        with torch.inference_mode():
            from tqdm import tqdm
            for imgs, paths_list in tqdm(dataloader):
                if imgs.nelement() == 0: continue
                preds = self.forward(imgs.to(self.device))
                for path, p_idx in zip(paths_list, preds):
                    results.append({
                        'filename': os.path.splitext(os.path.basename(path))[0],
                        'predicted_label': self.label_int_to_str(p_idx)
                    })
        return results

    def forward(self, batch):
        with torch.inference_mode():
            logits = self.model(batch)
            probs = F.softmax(logits, dim=1).cpu().numpy()
            return np.argmax(np.matmul(probs, self.hierarchy_places3), axis=1).tolist()

    def label_int_to_str(self, idx):
        return {0: 'Indoor', 1: 'Natural', 2: 'Urban'}.get(idx, 'Unknown')
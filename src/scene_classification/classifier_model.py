import os
import csv
import numpy as np
import torch
import torchvision
import torchvision.transforms as tfm
import torch.nn.functional as F
from PIL import Image
from downloader import download_scene_hierarchy_file, download_pretrained_on_places

class SceneClassifier(torch.nn.Module):
    def __init__(self, scene_hierarchy_file='scene_hierarchy_places365.csv', model_name="resnet50"):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using Device: {self.device}")

        if not os.path.exists(scene_hierarchy_file):
            download_scene_hierarchy_file()
        if not os.path.exists(f"{model_name}_places365.pth.tar"):
            download_pretrained_on_places(model_name)

        hierarchy_places3 = []
        with open(scene_hierarchy_file, 'r', encoding='utf-8') as csvfile:
            content = csv.reader(csvfile, delimiter=',')
            next(content); next(content)
            for line in content:
                hierarchy_places3.append(line[1:4])

        hierarchy_places3 = np.asarray(hierarchy_places3, dtype=float)
        sum_rows = np.sum(hierarchy_places3, axis=1)
        sum_rows[sum_rows == 0] = 1.0
        self.hierarchy_places3 = hierarchy_places3 / np.expand_dims(sum_rows, axis=-1)

        self.model = torchvision.models.resnet50(weights=None)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 365)

        checkpoint = torch.load(f"{model_name}_places365.pth.tar", map_location='cpu', weights_only=False)
        state_dict = {k.replace("module.", ""): v for k, v in checkpoint["state_dict"].items()}
        self.model.load_state_dict(state_dict)

        self.model.to(self.device)
        self.model.eval()

        self.transform = tfm.Compose([
            tfm.Resize((256, 256)),
            tfm.CenterCrop(224),
            tfm.ToTensor(),
            tfm.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def classify_one_image(self, image_path):
        """Classifies one image and returns its human-readable label."""
        if not os.path.exists(image_path):
            return "File not found"
        img = Image.open(image_path).convert('RGB')
        tensor = self.transform(img).unsqueeze(0).to(self.device)
        prediction_idx = self.forward(tensor)
        return self.label_int_to_str(prediction_idx[0])

    def forward(self, batch):
        with torch.inference_mode():
            logits = self.model(batch)
            scene_probs = F.softmax(logits, dim=1).cpu().numpy()
            places_prob = np.matmul(scene_probs, self.hierarchy_places3)
            scene_label_int = np.argmax(places_prob, axis=1)
        return scene_label_int.tolist()

    def label_int_to_str(self, scene_label_int):
        if scene_label_int == 0: return 'Indoor'
        elif scene_label_int == 1: return 'Natural'
        elif scene_label_int == 2: return 'Urban'
        return 'Unknown'
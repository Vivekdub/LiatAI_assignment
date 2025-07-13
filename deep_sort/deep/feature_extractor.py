import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../deep-person-reid")))

from torchreid.utils import load_pretrained_weights
from torchreid.models.osnet import osnet_x0_25

class Extractor:
    def __init__(self, model_path, use_cuda=True):
        self.device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")

        self.model = osnet_x0_25(num_classes=1000)
        load_pretrained_weights(self.model, model_path)
        self.model.eval().to(self.device)

        self.size = (256, 128)
        self.transform = T.Compose([
            T.Resize(self.size),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def _preprocess(self, crops):
        tensors = []
        for crop in crops:
            if isinstance(crop, np.ndarray):
                crop = Image.fromarray(crop[..., ::-1])
            tensors.append(self.transform(crop).unsqueeze(0))
        return torch.cat(tensors, dim=0).to(self.device) if tensors else torch.empty((0, 3, *self.size)).to(self.device)

    def __call__(self, frame, bboxes):
        crops = []
        h, w, _ = frame.shape
        for box in bboxes:
            x, y, bw, bh = [int(i) for i in box]
            x1 = max(int(x - bw / 2), 0)
            y1 = max(int(y - bh / 2), 0)
            x2 = min(int(x + bw / 2), w)
            y2 = min(int(y + bh / 2), h)

            if x2 > x1 and y2 > y1:
                crop = frame[y1:y2, x1:x2]
                crops.append(crop)

        if not crops:
            return np.array([])

        with torch.no_grad():
            inputs = self._preprocess(crops)
            outputs = self.model(inputs)
        return outputs.cpu().numpy()

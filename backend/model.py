import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
from typing import Tuple, Dict, Any

class EmotionModel:
    """
    Inference wrapper. Currently returns dummy predictions so you can
    validate the pipeline. Swap `predict_logits` with a trained model later.
    """

    def __init__(self, labels=None, image_size: int = 224, device: str = "cpu"):
        self.labels = labels or ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
        self.num_classes = len(self.labels)
        self.device = torch.device(device)

        # Preprocessing to make inputs consistent; keep this when you add a real model.
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),                         # (C,H,W) in [0,1]
            # Optional normalization; uncomment to match common ImageNet models.
            # transforms.Normalize(mean=[0.485, 0.456, 0.406],
            #                      std=[0.229, 0.224, 0.225]),
        ])

        # Placeholder: if you add a real model, move it to device and set eval()
        # self.model = ... ; self.model.to(self.device).eval()

    def preprocess_np_bgr(self, crop_bgr: np.ndarray) -> torch.Tensor:
        """Convert OpenCV BGR crop (H,W,3) -> transformed batch tensor (1,C,H,W)."""
        rgb = crop_bgr[:, :, ::-1]  # BGR -> RGB
        pil = Image.fromarray(rgb)
        tensor = self.transform(pil).unsqueeze(0).to(self.device)
        return tensor

    def predict_logits(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Dummy logits for now (random). Replace with: self.model(batch).
        Returns shape (B, num_classes).
        """
        # NOTE: remove randomness if you want deterministic outputs:
        # torch.manual_seed(0)
        B = batch.shape[0]
        logits = torch.randn(B, self.num_classes, device=self.device)
        return logits

    def predict_on_crop(self, crop_bgr: np.ndarray) -> Dict[str, Any]:
        """
        Returns {"label": str, "score": float} for a single face crop.
        """
        x = self.preprocess_np_bgr(crop_bgr)
        logits = self.predict_logits(x)
        probs = torch.softmax(logits, dim=1)[0]
        idx = int(torch.argmax(probs).item())
        return {
            "label": self.labels[idx],
            "score": float(probs[idx].item())
        }

    def get_labels(self):
        return self.labels

import torch 
import torch.nn as nn
import torchvision.models as models


#create a wrapper for Resnet18 image classification model
class EmoModel: 
    def __init__(self, num_of_classes=7):
        self.model = models.resnet18(pretrained=True)
        self.model.fc= nn.Linear(self.model.fc.in_features, num_of_classes)
        self.model.eval()
        
        def predict(self, face_tensor):
            with torch.no_grad():
                output = self.mode(face_tensor)
                _, pred = torch.max(output, 1)
            return pred.item
        
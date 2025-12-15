import torch
import torch.nn as nn
from torchvision import models

class SkinCancerNet(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.base = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
        in_feats = self.base.classifier[1].in_features
        self.base.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_feats, num_classes)
        )

    def forward(self, x):
        return self.base(x)

def load_model(weights_path="best_model.pth", device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = SkinCancerNet()
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

# model.py
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import VGG16_Weights

class CSRNet(nn.Module):
    def __init__(self, load_weights=False):
        super().__init__()
        # Frontend: VGG16 (use layers up to conv3 block so output has 256 channels)
        weights = VGG16_Weights.IMAGENET1K_V1 if load_weights else None
        vgg_features = models.vgg16(weights=weights).features
        # take the first 16 modules (up to the end of the conv3 block)
        self.frontend = nn.Sequential(*list(vgg_features)[:16])

        # Backend: Dilated convolutions
        self.backend = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=2, dilation=2), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=2, dilation=2), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=2, dilation=2), nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, 3, padding=2, dilation=2), nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=2, dilation=2), nn.ReLU(inplace=True),
            nn.Conv2d(128, 64,  3, padding=2, dilation=2), nn.ReLU(inplace=True),
            nn.Conv2d(64,  1,   1)  # Output: density map
        )

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        return x
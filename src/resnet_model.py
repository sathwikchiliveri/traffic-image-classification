"""
resnet_model.py

Transfer Learning using ResNet18
for Traffic Image Classification
"""

import torch
import torch.nn as nn
from torchvision import models


class TrafficResNet(nn.Module):

    def __init__(self, num_classes=5):

        super(TrafficResNet, self).__init__()

        # Load pretrained ResNet18
        self.model = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT
        )

        # Freeze pretrained layers
        for param in self.model.parameters():
            param.requires_grad = False

        # Replace classifier
        in_features = self.model.fc.in_features

        self.model.fc = nn.Sequential(

            nn.Linear(in_features, 256),

            nn.ReLU(),

            nn.Dropout(0.5),

            nn.Linear(256, num_classes)

        )

    def forward(self, x):

        return self.model(x)


if __name__ == "__main__":

    model = TrafficResNet()

    dummy = torch.randn(1, 3, 224, 224)

    output = model(dummy)

    print(model)

    print("\nOutput Shape:", output.shape)
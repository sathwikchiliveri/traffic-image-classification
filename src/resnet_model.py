"""
resnet_model.py

Transfer Learning using ResNet18
for Traffic Image Classification
"""

import torch
import torch.nn as nn
from torchvision import models


class TrafficResNet(nn.Module):
    """
    ResNet18 Transfer Learning Model
    """

    def __init__(self, num_classes=5):

        super(TrafficResNet, self).__init__()

        # =====================================================
        # Load Pretrained ResNet18
        # =====================================================

        self.model = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT
        )

        # =====================================================
        # Freeze Entire Network
        # =====================================================

        for param in self.model.parameters():
            param.requires_grad = False

        # =====================================================
        # Unfreeze Last Residual Block (layer4)
        # =====================================================

        for param in self.model.layer4.parameters():
            param.requires_grad = True

        # =====================================================
        # Replace Final Classifier
        # =====================================================

        in_features = self.model.fc.in_features

        self.model.fc = nn.Sequential(

            nn.Linear(in_features, 256),

            nn.ReLU(),

            nn.Dropout(0.5),

            nn.Linear(256, num_classes)

        )

        # Make sure classifier is trainable

        for param in self.model.fc.parameters():
            param.requires_grad = True

    # =====================================================
    # Forward Pass
    # =====================================================

    def forward(self, x):

        return self.model(x)


# =====================================================
# Test
# =====================================================

if __name__ == "__main__":

    model = TrafficResNet()

    dummy_input = torch.randn(1, 3, 224, 224)

    output = model(dummy_input)

    print("=" * 60)
    print("ResNet18 Loaded Successfully")
    print("=" * 60)

    print(model)

    print("\nOutput Shape:", output.shape)

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    print(f"\nTrainable Parameters : {trainable_params:,}")
    print(f"Total Parameters     : {total_params:,}")
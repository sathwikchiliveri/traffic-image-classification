"""
model.py

Custom CNN Architecture for
Traffic Image Classification
"""

import torch
import torch.nn as nn


class TrafficCNN(nn.Module):
    """
    Custom Convolutional Neural Network
    """

    def __init__(self, num_classes=5):
        super(TrafficCNN, self).__init__()

        # =====================================================
        # Feature Extractor
        # =====================================================

        self.features = nn.Sequential(

            # Block 1
            nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 2
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 3
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # =====================================================
        # Classifier
        # =====================================================

        self.classifier = nn.Sequential(

            nn.AdaptiveAvgPool2d((1, 1)),

            nn.Flatten(),

            nn.Linear(128, 64),

            nn.ReLU(),

            nn.Dropout(0.5),

            nn.Linear(64, num_classes)
        )

    def forward(self, x):

        x = self.features(x)

        x = self.classifier(x)

        return x


if __name__ == "__main__":

    model = TrafficCNN()

    dummy_input = torch.randn(1, 3, 128, 128)

    output = model(dummy_input)

    print(model)

    print("\nOutput Shape:", output.shape)
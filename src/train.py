"""
train.py

Training script for Traffic Image Classification
"""

import torch
import torch.nn as nn
import torch.optim as optim

from dataset import load_datasets, create_dataloaders
from model import TrafficCNN
from config import *


def main():

    print("=" * 60)
    print("Traffic Image Classification")
    print("=" * 60)

    # ==================================
    # Load Dataset
    # ==================================

    print("\nLoading Datasets...")

    train_dataset, valid_dataset, test_dataset = load_datasets(
        TRAIN_DIR,
        VALID_DIR,
        TEST_DIR
    )

    print("Datasets Loaded Successfully!")

    print(f"Training Images   : {len(train_dataset)}")
    print(f"Validation Images : {len(valid_dataset)}")
    print(f"Testing Images    : {len(test_dataset)}")

    # ==================================
    # Create DataLoaders
    # ==================================

    print("\nCreating DataLoaders...")

    train_loader, valid_loader, test_loader = create_dataloaders(
        train_dataset,
        valid_dataset,
        test_dataset
    )

    print("DataLoaders Created Successfully!")

    print(f"Training Batches   : {len(train_loader)}")
    print(f"Validation Batches : {len(valid_loader)}")
    print(f"Testing Batches    : {len(test_loader)}")

    # ==================================
    # Load Model
    # ==================================

    print("\nLoading CNN Model...")

    model = TrafficCNN()

    model = model.to(DEVICE)

    print("CNN Loaded Successfully!")

    print(f"Running on : {DEVICE}")

    # ==================================
    # Loss Function
    # ==================================

    criterion = nn.CrossEntropyLoss()

    # ==================================
    # Optimizer
    # ==================================

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    print("Loss Function : CrossEntropyLoss")
    print("Optimizer     : Adam")

    # ==================================
    # Test Forward Pass
    # ==================================

    print("\nTesting Forward Pass...")

    images, labels = next(iter(train_loader))

    images = images.to(DEVICE)
    labels = labels.to(DEVICE)

    outputs = model(images)

    print(f"Input Shape  : {images.shape}")
    print(f"Labels Shape : {labels.shape}")
    print(f"Output Shape : {outputs.shape}")

    print("\nPipeline Working Successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
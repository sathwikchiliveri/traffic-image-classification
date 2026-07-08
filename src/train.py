"""
test.py

Quick test to verify that the dataset,
dataloader, and model are working correctly.
"""

import torch

from dataset import load_datasets, create_dataloaders
from model import TrafficCNN

from config import TRAIN_DIR, VALID_DIR, TEST_DIR


def main():

    # Load datasets
    train_dataset, valid_dataset, test_dataset = load_datasets(
        TRAIN_DIR,
        VALID_DIR,
        TEST_DIR
    )

    # Create dataloaders
    train_loader, _, _ = create_dataloaders(
        train_dataset,
        valid_dataset,
        test_dataset
    )

    # Create model
    model = TrafficCNN()

    # Get one batch
    images, labels = next(iter(train_loader))

    print("=" * 50)
    print("Input Shape :", images.shape)
    print("Labels Shape:", labels.shape)

    outputs = model(images)

    print("Output Shape:", outputs.shape)

    print("=" * 50)
    print("Classes:", train_dataset.classes)


if __name__ == "__main__":
    main()
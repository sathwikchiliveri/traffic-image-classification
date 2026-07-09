"""
train.py

Training script for Traffic Image Classification
"""

from turtle import forward

import torch
import torch.nn as nn
import torch.optim as optim

from dataset import (
    load_datasets,
    create_dataloaders
)

from model import TrafficCNN

from config import *

def main():

    print("="*50)
    print("Traffic Image Classification")
    print("="*50)

if __name__ == "__main__":
    main()
    train_dataset, valid_dataset, test_dataset = load_datasets(
    TRAIN_DIR,
    VALID_DIR,
    TEST_DIR
)

train_loader, valid_loader, test_loader = create_dataloaders(
    train_dataset,
    valid_dataset,
    test_dataset
)

print(f"\nTraining Images   : {len(train_dataset)}")
print(f"Validation Images : {len(valid_dataset)}")
print(f"Testing Images    : {len(test_dataset)}")
model = TrafficCNN()
print("\nModel Loaded Successfully!")
model.to(DEVICE)

print(f"Running on : {DEVICE}")
images, labels = next(iter(train_loader))

images = images.to(DEVICE)

outputs = model(images)

print("\nInput Shape :", images.shape)
print("Output Shape:", outputs.shape)
##test one forward pass##
images, labels = next(iter(train_loader))

images = images.to(DEVICE)

outputs = model(images)

print("\nInput Shape :", images.shape)
print("Output Shape:", outputs.shape)
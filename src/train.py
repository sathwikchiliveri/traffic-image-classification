"""
train.py

Training script for Traffic Image Classification
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim

from dataset import load_datasets, create_dataloaders
from model import TrafficCNN
from config import *


# =====================================================
# Train One Epoch
# =====================================================

def train_one_epoch(model, train_loader, criterion, optimizer, device):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(train_loader):

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()
        if (batch_idx + 1) % 20 == 0:
         print(
        f"Batch [{batch_idx + 1}/{len(train_loader)}] "
        f"Loss: {loss.item():.4f}"
         )

    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = 100 * correct / total

    return epoch_loss, epoch_accuracy


# =====================================================
# Validation
# =====================================================

def validate_one_epoch(model, valid_loader, criterion, device):

    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in valid_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)

            correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(valid_loader)
    epoch_accuracy = 100 * correct / total

    return epoch_loss, epoch_accuracy


# =====================================================
# Save Model
# =====================================================

def save_model(model, path):

    folder = os.path.dirname(path)

    os.makedirs(folder, exist_ok=True)

    torch.save(model.state_dict(), path)

    print(f"\nModel saved successfully!")
    print(path)


# =====================================================
# Main Function
# =====================================================

def main():

    print("=" * 60)
    print("Traffic Image Classification")
    print("=" * 60)

    print("\nLoading datasets...")

    train_dataset, valid_dataset, test_dataset = load_datasets(
        TRAIN_DIR,
        VALID_DIR,
        TEST_DIR
    )

    print("Datasets Loaded Successfully!")

    print(f"Training Images   : {len(train_dataset)}")
    print(f"Validation Images : {len(valid_dataset)}")
    print(f"Testing Images    : {len(test_dataset)}")

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

    print("\nLoading CNN Model...")

    model = TrafficCNN()

    model = model.to(DEVICE)

    print("CNN Loaded Successfully!")

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    print("Loss Function : CrossEntropyLoss")
    print("Optimizer     : Adam")

    best_accuracy = 0.0

    train_losses = []
    train_accuracies = []

    valid_losses = []
    valid_accuracies = []
        # =====================================================
    # Training Loop
    # =====================================================

    print("\nStarting Training...\n")

    for epoch in range(EPOCHS):

        print("=" * 60)
        print(f"Epoch [{epoch + 1}/{EPOCHS}]")
        print("=" * 60)

        # -------------------------------
        # Train
        # -------------------------------

        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            DEVICE
        )

        # -------------------------------
        # Validation
        # -------------------------------

        valid_loss, valid_accuracy = validate_one_epoch(
            model,
            valid_loader,
            criterion,
            DEVICE
        )

        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        valid_losses.append(valid_loss)
        valid_accuracies.append(valid_accuracy)

        print(f"Train Loss     : {train_loss:.4f}")
        print(f"Train Accuracy : {train_accuracy:.2f}%")

        print(f"Valid Loss     : {valid_loss:.4f}")
        print(f"Valid Accuracy : {valid_accuracy:.2f}%")

        # -------------------------------
        # Save Best Model
        # -------------------------------

        if valid_accuracy > best_accuracy:

            best_accuracy = valid_accuracy

            save_model(
                model,
                MODEL_PATH
            )

            print("Best model updated!")

        print()

    # =====================================================
    # Training Summary
    # =====================================================

    print("=" * 60)
    print("Training Finished")
    print("=" * 60)

    print(f"Best Validation Accuracy : {best_accuracy:.2f}%")

    print("\nTraining Complete!")

    return {
        "train_loss": train_losses,
        "train_accuracy": train_accuracies,
        "valid_loss": valid_losses,
        "valid_accuracy": valid_accuracies,
    }


# =====================================================
# Entry Point
# =====================================================

if __name__ == "__main__":
    main()
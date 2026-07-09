"""
train.py

Traffic Image Classification
Training Script
"""

import csv
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim

from dataset_resnet import create_dataloaders
from resnet_model import TrafficResNet
from config import *
RESNET_MODEL_PATH = MODEL_DIR / "traffic_resnet18.pth"

# ==========================================================
# Calculate Class Weights
# ==========================================================

def get_class_weights(train_dataset):

    class_counts = Counter(train_dataset.targets)

    print("\nDataset Distribution")
    print("-" * 40)

    for class_id, count in sorted(class_counts.items()):
        print(f"Class {class_id} : {count}")

    total_samples = len(train_dataset)
    num_classes = len(class_counts)

    weights = []

    for class_id in range(num_classes):

        weight = total_samples / (
            num_classes * class_counts[class_id]
        )

        weights.append(weight)

    weights = torch.tensor(
        weights,
        dtype=torch.float32
    )

    weights = weights.to(DEVICE)

    print("\nClass Weights")
    print("-" * 40)

    for class_id, weight in enumerate(weights):
        print(f"Class {class_id} : {weight:.3f}")

    return weights


# ==========================================================
# Save Training History
# ==========================================================

def save_training_history(
    train_losses,
    train_accuracies,
    valid_losses,
    valid_accuracies
):

    LOG_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    history_file = LOG_DIR / "training_history.csv"

    with open(
        history_file,
        "w",
        newline=""
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "Epoch",
            "Train Loss",
            "Train Accuracy",
            "Validation Loss",
            "Validation Accuracy"
        ])

        for epoch in range(len(train_losses)):

            writer.writerow([
                epoch + 1,
                train_losses[epoch],
                train_accuracies[epoch],
                valid_losses[epoch],
                valid_accuracies[epoch]
            ])

    print("\nTraining history saved.")

    print(history_file)


# ==========================================================
# Train One Epoch
# ==========================================================

def train_one_epoch(
    model,
    train_loader,
    criterion,
    optimizer
):

    model.train()

    running_loss = 0.0

    correct = 0

    total = 0

    for batch_idx, (images, labels) in enumerate(train_loader):

        images = images.to(DEVICE)

        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(
            outputs,
            1
        )

        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()

        if (batch_idx + 1) % 20 == 0:

            print(
                f"Batch [{batch_idx+1}/{len(train_loader)}] "
                f"Loss : {loss.item():.4f}"
            )

    epoch_loss = running_loss / len(train_loader)

    epoch_accuracy = 100 * correct / total

    return epoch_loss, epoch_accuracy
# ==========================================================
# Validate One Epoch
# ==========================================================

def validate_one_epoch(
    model,
    valid_loader,
    criterion
):

    model.eval()

    running_loss = 0.0

    correct = 0

    total = 0

    with torch.no_grad():

        for images, labels in valid_loader:

            images = images.to(DEVICE)

            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            running_loss += loss.item()

            _, predicted = torch.max(
                outputs,
                1
            )

            total += labels.size(0)

            correct += (
                predicted == labels
            ).sum().item()

    epoch_loss = running_loss / len(valid_loader)

    epoch_accuracy = 100 * correct / total

    return epoch_loss, epoch_accuracy


# ==========================================================
# Save Best Model
# ==========================================================

def save_model(model, path):

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    torch.save(
        model.state_dict(),
        path
    )

    print("\nModel Saved Successfully!")
    print(path)

    print("\nBest model saved successfully!")

    print(RESNET_MODEL_PATH)


# ==========================================================
# Main
# ==========================================================

def main():

    print("=" * 60)
    print("Traffic Image Classification")
    print("=" * 60)

    # ------------------------------------------------------
    # Load Dataset
    # ------------------------------------------------------
(
    train_loader,
    valid_loader,
    test_loader,
    train_dataset,
    valid_dataset,
    test_dataset
) = create_dataloaders()

    # ------------------------------------------------------
    # DataLoaders
    # ------------------------------------------------------

(
    train_loader,
    valid_loader,
    test_loader,
    train_dataset,
    valid_dataset,
    test_dataset
) = create_dataloaders()

print("DataLoaders Created Successfully!")

print(f"Training Batches   : {len(train_loader)}")
print(f"Validation Batches : {len(valid_loader)}")
print(f"Testing Batches    : {len(test_loader)}")

    # ------------------------------------------------------
    # Model
    # ------------------------------------------------------

print("\nLoading CNN Model...")

model = TrafficResNet()

model = model.to(DEVICE)
print("CNN Loaded Successfully!")

    # ------------------------------------------------------
    # Weighted Loss
    # ------------------------------------------------------

class_weights = get_class_weights(
        train_dataset
    )

criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=0.0001
)

print("\nLoss Function : Weighted CrossEntropyLoss")

print("Optimizer     : Adam")

best_accuracy = 0.0

train_losses = []

train_accuracies = []

valid_losses = []

valid_accuracies = []
        # ==========================================================
    # Training Loop
    # ==========================================================

print("\nStarting Training...\n")

for epoch in range(EPOCHS):

        print("=" * 60)
        print(f"Epoch [{epoch + 1}/{EPOCHS}]")
        print("=" * 60)

        # ---------------------------
        # Train
        # ---------------------------

        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer
        )

        # ---------------------------
        # Validation
        # ---------------------------

        valid_loss, valid_accuracy = validate_one_epoch(
            model,
            valid_loader,
            criterion
        )

        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        valid_losses.append(valid_loss)
        valid_accuracies.append(valid_accuracy)

        print("\nEpoch Summary")
        print("-" * 40)

        print(f"Training Loss       : {train_loss:.4f}")
        print(f"Training Accuracy   : {train_accuracy:.2f}%")

        print(f"Validation Loss     : {valid_loss:.4f}")
        print(f"Validation Accuracy : {valid_accuracy:.2f}%")

        # ---------------------------
        # Save Best Model
        # ---------------------------

        if valid_accuracy > best_accuracy:

            best_accuracy = valid_accuracy

            save_model(
          model,
          MODEL_DIR / "traffic_resnet18.pth"
)

        else:

            print("Validation accuracy did not improve.")

        print()

    # ==========================================================
    # Save Training History
    # ==========================================================

save_training_history(
        train_losses,
        train_accuracies,
        valid_losses,
        valid_accuracies
    )

    # ==========================================================
    # Final Summary
    # ==========================================================

print("=" * 60)
print("Training Finished")
print("=" * 60)

print(f"Best Validation Accuracy : {best_accuracy:.2f}%")

print("\nTraining History")

for epoch in range(len(train_losses)):

        print(
            f"Epoch {epoch + 1:02d} | "
            f"Train Loss: {train_losses[epoch]:.4f} | "
            f"Train Acc: {train_accuracies[epoch]:.2f}% | "
            f"Valid Loss: {valid_losses[epoch]:.4f} | "
            f"Valid Acc: {valid_accuracies[epoch]:.2f}%"
        )

print("\nTraining completed successfully!")



print("Model saved at:")

print(MODEL_DIR / "traffic_resnet18.pth")
print(f"\nTraining history saved at:\n{LOG_DIR / 'training_history.csv'}")


# ==========================================================
# Entry Point
# ==========================================================

if __name__ == "__main__":
    main()
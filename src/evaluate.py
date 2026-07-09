"""
evaluate.py

Evaluate the trained Traffic CNN
on the testing dataset.
"""

import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

from dataset import load_datasets, create_dataloaders
from model import TrafficCNN
from config import *


def evaluate():

    print("=" * 60)
    print("Traffic Image Classification Evaluation")
    print("=" * 60)

    # -----------------------------------------
    # Load Dataset
    # -----------------------------------------

    _, _, test_dataset = load_datasets(
        TRAIN_DIR,
        VALID_DIR,
        TEST_DIR
    )

    _, _, test_loader = create_dataloaders(
        test_dataset,
        test_dataset,
        test_dataset
    )

    # -----------------------------------------
    # Load Model
    # -----------------------------------------

    model = TrafficCNN()

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=DEVICE
        )
    )

    model.to(DEVICE)

    model.eval()

    print("\nModel Loaded Successfully!")

    # -----------------------------------------
    # Evaluation
    # -----------------------------------------

    all_predictions = []
    all_labels = []

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(DEVICE)

            outputs = model(images)

            _, predicted = torch.max(outputs, 1)

            all_predictions.extend(predicted.cpu().numpy())

            all_labels.extend(labels.numpy())

    # -----------------------------------------
    # Metrics
    # -----------------------------------------

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    print("\nTest Accuracy")
    print("-" * 30)
    print(f"{accuracy*100:.2f}%")

    print("\nConfusion Matrix")
    print("-" * 30)
    print(
        confusion_matrix(
            all_labels,
            all_predictions
        )
    )

    print("\nClassification Report")
    print("-" * 30)
    print(
        classification_report(
            all_labels,
            all_predictions,
            target_names=test_dataset.classes
        )
    )


if __name__ == "__main__":
    evaluate()
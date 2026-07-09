"""
evaluate.py

Evaluate the trained Traffic Image Classification model.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from dataset import load_datasets, create_dataloaders
from model import TrafficCNN
from config import *


# ==========================================================
# Save Confusion Matrix
# ==========================================================

def save_confusion_matrix(cm, class_names):

    GRAPH_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    plt.figure(figsize=(8, 6))

    plt.imshow(
        cm,
        interpolation="nearest",
        cmap="Blues"
    )

    plt.title("Confusion Matrix")

    plt.colorbar()

    tick_marks = np.arange(len(class_names))

    plt.xticks(
        tick_marks,
        class_names,
        rotation=45
    )

    plt.yticks(
        tick_marks,
        class_names
    )

    threshold = cm.max() / 2

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):

            plt.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > threshold else "black"
            )

    plt.ylabel("Actual")

    plt.xlabel("Predicted")

    plt.tight_layout()

    output_path = GRAPH_DIR / "confusion_matrix.png"

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"\nConfusion Matrix saved successfully!")
    print(output_path)


# ==========================================================
# Evaluation
# ==========================================================

def evaluate():

    print("=" * 60)
    print("Traffic Image Classification Evaluation")
    print("=" * 60)

    # ------------------------------------------------------
    # Load Dataset
    # ------------------------------------------------------

    train_dataset, valid_dataset, test_dataset = load_datasets(
        TRAIN_DIR,
        VALID_DIR,
        TEST_DIR
    )

    _, _, test_loader = create_dataloaders(
        train_dataset,
        valid_dataset,
        test_dataset
    )

    # ------------------------------------------------------
    # Load Model
    # ------------------------------------------------------

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

    # ------------------------------------------------------
    # Prediction
    # ------------------------------------------------------

    all_predictions = []

    all_labels = []

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(DEVICE)

            outputs = model(images)

            _, predicted = torch.max(outputs, 1)

            all_predictions.extend(
                predicted.cpu().numpy()
            )

            all_labels.extend(
                labels.numpy()
            )

    # ------------------------------------------------------
    # Accuracy
    # ------------------------------------------------------

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    print("\nTest Accuracy")
    print("-" * 30)

    print(f"{accuracy * 100:.2f}%")

    # ------------------------------------------------------
    # Confusion Matrix
    # ------------------------------------------------------

    cm = confusion_matrix(
        all_labels,
        all_predictions
    )

    print("\nConfusion Matrix")
    print("-" * 30)

    print(cm)

    save_confusion_matrix(
        cm,
        test_dataset.classes
    )

    # ------------------------------------------------------
    # Classification Report
    # ------------------------------------------------------

    print("\nClassification Report")
    print("-" * 30)

    print(
        classification_report(
            all_labels,
            all_predictions,
            target_names=test_dataset.classes
        )
    )


# ==========================================================
# Entry Point
# ==========================================================

if __name__ == "__main__":
    evaluate()
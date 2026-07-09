"""
graphs.py

Generate training graphs from training_history.csv
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from config import LOG_DIR, GRAPH_DIR


def plot_graphs():

    history_file = LOG_DIR / "training_history.csv"

    if not history_file.exists():
        print("Training history not found!")
        print(history_file)
        return

    GRAPH_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    history = pd.read_csv(history_file)

    # ----------------------------------
    # Loss Curve
    # ----------------------------------

    plt.figure(figsize=(8, 5))

    plt.plot(
        history["Epoch"],
        history["Train Loss"],
        label="Train Loss",
        linewidth=2
    )

    plt.plot(
        history["Epoch"],
        history["Validation Loss"],
        label="Validation Loss",
        linewidth=2
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.grid(True)
    plt.legend()

    loss_path = GRAPH_DIR / "loss_curve.png"

    plt.savefig(loss_path, dpi=300)

    plt.close()

    # ----------------------------------
    # Accuracy Curve
    # ----------------------------------

    plt.figure(figsize=(8, 5))

    plt.plot(
        history["Epoch"],
        history["Train Accuracy"],
        label="Train Accuracy",
        linewidth=2
    )

    plt.plot(
        history["Epoch"],
        history["Validation Accuracy"],
        label="Validation Accuracy",
        linewidth=2
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Training vs Validation Accuracy")
    plt.grid(True)
    plt.legend()

    accuracy_path = GRAPH_DIR / "accuracy_curve.png"

    plt.savefig(accuracy_path, dpi=300)

    plt.close()

    print("=" * 60)
    print("Graphs Generated Successfully!")
    print("=" * 60)

    print(f"Loss Curve     : {loss_path}")
    print(f"Accuracy Curve : {accuracy_path}")


if __name__ == "__main__":
    plot_graphs()
"""
config.py

Stores all configurable parameters
used throughout the project.
"""

from pathlib import Path

# ======================================================
# Project Root
# ======================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ======================================================
# Dataset
# ======================================================

TRAIN_DIR = PROJECT_ROOT / "Data" / "training"
VALID_DIR = PROJECT_ROOT / "Data" / "validation"
TEST_DIR = PROJECT_ROOT / "Data" / "testing"

# ======================================================
# Image
# ======================================================

IMAGE_SIZE = (128, 128)

# ======================================================
# Model
# ======================================================

NUM_CLASSES = 5

# ======================================================
# Training
# ======================================================

BATCH_SIZE = 8

LEARNING_RATE = 0.001

EPOCHS = 20

RANDOM_SEED = 42

# ======================================================
# Hardware
# ======================================================

DEVICE = "cpu"

# ======================================================
# Outputs
# ======================================================

OUTPUT_DIR = PROJECT_ROOT / "outputs"

MODEL_DIR = OUTPUT_DIR / "models"

GRAPH_DIR = OUTPUT_DIR / "graphs"

LOG_DIR = OUTPUT_DIR / "logs"

PREDICTION_DIR = OUTPUT_DIR / "predictions"

# ======================================================
# Checkpoints
# ======================================================

MODEL_PATH = MODEL_DIR / "traffic_classifier.pth"
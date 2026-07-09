"""
config.py

Stores all configurable parameters
used throughout the project.
"""

# ===============================
# Dataset
# ===============================

TRAIN_DIR = "Data/training"
VALID_DIR = "Data/validation"
TEST_DIR = "Data/testing"

# ===============================
# Image
# ===============================

IMAGE_SIZE = (224, 224)

# ===============================
# Model
# ===============================

NUM_CLASSES = 5

# ===============================
# Training
# ===============================

BATCH_SIZE = 16

LEARNING_RATE = 0.001

EPOCHS = 20

RANDOM_SEED = 42

# ===============================
# Hardware
# ===============================

DEVICE = "cpu"

# ===============================
# Checkpoints
# ===============================

MODEL_PATH = "../outputs/models/traffic_classifier.pth"
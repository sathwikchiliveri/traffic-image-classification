"""
dataset.py

Handles:
1. Image Transformations
2. Dataset Loading
3. DataLoader Creation
"""

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from config import IMAGE_SIZE, BATCH_SIZE

# ============================
# Configuration
# ============================

#IMAGE_SIZE = (224, 224)
#BATCH_SIZE = 32


# ============================
# Image Transformations
# ============================

train_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),

    # Data Augmentation
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


test_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================
# Dataset Loader
# ============================

def load_datasets(train_path, valid_path, test_path):

    train_dataset = datasets.ImageFolder(
        root=train_path,
        transform=train_transform
    )

    valid_dataset = datasets.ImageFolder(
        root=valid_path,
        transform=test_transform
    )

    test_dataset = datasets.ImageFolder(
        root=test_path,
        transform=test_transform
    )

    return train_dataset, valid_dataset, test_dataset


# ============================
# DataLoader
# ============================

def create_dataloaders(train_dataset,
                       valid_dataset,
                       test_dataset):

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    return train_loader, valid_loader, test_loader
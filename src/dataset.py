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


# =====================================================
# Image Transformations
# =====================================================

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


valid_test_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# =====================================================
# Load Datasets
# =====================================================

def load_datasets(train_path, valid_path, test_path):
    """
    Loads training, validation and testing datasets.
    """

    train_dataset = datasets.ImageFolder(
        root=train_path,
        transform=train_transform
    )

    valid_dataset = datasets.ImageFolder(
        root=valid_path,
        transform=valid_test_transform
    )

    test_dataset = datasets.ImageFolder(
        root=test_path,
        transform=valid_test_transform
    )

    return train_dataset, valid_dataset, test_dataset


# =====================================================
# Create DataLoaders
# =====================================================

def create_dataloaders(train_dataset,
                       valid_dataset,
                       test_dataset):
    """
    Creates DataLoaders for training, validation and testing.
    Optimized for Windows + 8GB RAM.
    """

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )

    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    return train_loader, valid_loader, test_loader


# =====================================================
# Test
# =====================================================

if __name__ == "__main__":

    from config import TRAIN_DIR, VALID_DIR, TEST_DIR

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

    print("=" * 50)
    print("Dataset Loaded Successfully")
    print("=" * 50)

    print(f"Training Images   : {len(train_dataset)}")
    print(f"Validation Images : {len(valid_dataset)}")
    print(f"Testing Images    : {len(test_dataset)}")

    print()

    print(f"Training Batches   : {len(train_loader)}")
    print(f"Validation Batches : {len(valid_loader)}")
    print(f"Testing Batches    : {len(test_loader)}")

    print("=" * 50)
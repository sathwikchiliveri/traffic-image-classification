"""
dataset_resnet.py

Dataset loader for ResNet18
Transfer Learning
"""

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from config import (
    TRAIN_DIR,
    VALID_DIR,
    TEST_DIR,
    BATCH_SIZE
)

# =====================================================
# Image Transformations
# =====================================================

train_transform = transforms.Compose([

    transforms.Resize((224, 224)),

    transforms.RandomHorizontalFlip(p=0.5),

    transforms.RandomRotation(10),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

])


valid_test_transform = transforms.Compose([

    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

])

# =====================================================
# Load Datasets
# =====================================================

def load_datasets():

    train_dataset = datasets.ImageFolder(
        root=TRAIN_DIR,
        transform=train_transform
    )

    valid_dataset = datasets.ImageFolder(
        root=VALID_DIR,
        transform=valid_test_transform
    )

    test_dataset = datasets.ImageFolder(
        root=TEST_DIR,
        transform=valid_test_transform
    )

    return train_dataset, valid_dataset, test_dataset


# =====================================================
# Create DataLoaders
# =====================================================

def create_dataloaders():

    train_dataset, valid_dataset, test_dataset = load_datasets()

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    return (
        train_loader,
        valid_loader,
        test_loader,
        train_dataset,
        valid_dataset,
        test_dataset
    )


# =====================================================
# Test
# =====================================================

if __name__ == "__main__":

    (
        train_loader,
        valid_loader,
        test_loader,
        train_dataset,
        valid_dataset,
        test_dataset
    ) = create_dataloaders()

    print("=" * 60)
    print("ResNet Dataset Loaded Successfully")
    print("=" * 60)

    print(f"Training Images   : {len(train_dataset)}")
    print(f"Validation Images : {len(valid_dataset)}")
    print(f"Testing Images    : {len(test_dataset)}")

    print()

    print(f"Training Batches   : {len(train_loader)}")
    print(f"Validation Batches : {len(valid_loader)}")
    print(f"Testing Batches    : {len(test_loader)}")
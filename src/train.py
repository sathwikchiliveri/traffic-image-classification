"""
train.py

Training script for Traffic Image Classification
"""

import torch
import torch.nn as nn
import torch.optim as optim

from dataset import (
    load_datasets,
    create_dataloaders
)

from model import TrafficCNN

from config import *

def main():

    print("="*50)
    print("Traffic Image Classification")
    print("="*50)

if __name__ == "__main__":
    main()
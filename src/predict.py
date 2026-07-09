"""
predict.py

Predict the class of a single traffic image.
"""

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from model import TrafficCNN
from config import *


# =====================================================
# Class Names
# =====================================================

CLASS_NAMES = [
    "Empty",
    "High",
    "Low",
    "Medium",
    "Traffic Jam"
]


# =====================================================
# Image Transform
# =====================================================

transform = transforms.Compose([

    transforms.Resize(IMAGE_SIZE),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

])


# =====================================================
# Load Model
# =====================================================

model = TrafficCNN()

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)

model.to(DEVICE)

model.eval()


# =====================================================
# Prediction Function
# =====================================================

def predict(image_path):

    image = Image.open(image_path).convert("RGB")

    image = transform(image)

    image = image.unsqueeze(0)

    image = image.to(DEVICE)

    with torch.no_grad():

        output = model(image)

        probabilities = F.softmax(output, dim=1)

        confidence, predicted = torch.max(probabilities, 1)

    print("=" * 50)

    print("Prediction Result")

    print("=" * 50)

    print(f"Image      : {image_path}")

    print(f"Prediction : {CLASS_NAMES[predicted.item()]}")

    print(f"Confidence : {confidence.item()*100:.2f}%")

    print("=" * 50)


# =====================================================
# Main
# =====================================================

if __name__ == "__main__":

    image_path = input("Enter image path : ")

    predict(image_path)
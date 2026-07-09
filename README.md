# 🚦 Traffic Image Classification using Deep Learning

A Computer Vision project that classifies traffic images into five traffic conditions using a Custom Convolutional Neural Network (CNN) built with PyTorch.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-red)
![OpenCV](https://img.shields.io/badge/Computer-Vision-green)
![License](https://img.shields.io/badge/License-MIT-yellow)
## 📌 Project Overview

This project implements a custom Convolutional Neural Network (CNN) for multi-class traffic image classification.

The model classifies traffic scenes into five categories:

- Empty Road
- High Traffic
- Low Traffic
- Medium Traffic
- Traffic Jam

The complete pipeline includes:

- Dataset preprocessing
- Data augmentation
- Custom CNN architecture
- Weighted CrossEntropy Loss
- Model training
- Evaluation
- Single image prediction
- Performance visualization
## 📂 Project Structure

```text
traffic-image-classification
│
├── Notebooks/
├── outputs/
│   └── graphs/
│       ├── accuracy_curve.png
│       ├── loss_curve.png
│       └── confusion_matrix.png
│
├── src/
│   ├── config.py
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   └── graphs.py
│
├── requirements.txt
├── README.md
└── LICENSE
```
## 🧠 Model Architecture

Custom CNN

Input (128×128×3)

↓

Conv2D (32 Filters)

↓

ReLU

↓

MaxPooling

↓

Conv2D (64 Filters)

↓

ReLU

↓

MaxPooling

↓

Conv2D (128 Filters)

↓

ReLU

↓

MaxPooling

↓

Adaptive Average Pooling

↓

Fully Connected Layer

↓

Dropout (0.5)

↓

Output Layer (5 Classes)
## ⚙️ Training Configuration

| Parameter | Value |
|-----------|------:|
| Epochs | 20 |
| Batch Size | 8 |
| Learning Rate | 0.001 |
| Optimizer | Adam |
| Loss Function | Weighted CrossEntropyLoss |
| Image Size | 128×128 |
## 📊 Results

| Metric | Value |
|--------|------:|
| Test Accuracy | **79.38%** |
| Classes | 5 |
| Framework | PyTorch |
## 📈 Training Accuracy

![Accuracy](outputs/graphs/accuracy_curve.png)

---

## 📉 Training Loss

![Loss](outputs/graphs/loss_curve.png)

---

## 🔲 Confusion Matrix

![Confusion Matrix](outputs/graphs/confusion_matrix.png)
## 🔍 Predict a Single Image

```bash
python src/predict.py
```

Example

```text
Prediction : High
Confidence : 73.74%
```
## 🚀 Installation

```bash
git clone https://github.com/sathwikchiliveri/traffic-image-classification.git

cd traffic-image-classification

pip install -r requirements.txt
```

# 🚦 Traffic Image Classification using Deep Learning

A PyTorch-based deep learning project for multi-class traffic image classification using a custom Convolutional Neural Network (CNN). The model classifies traffic scenes into five categories and is trained using weighted cross-entropy loss to address dataset imbalance.

---

## 📌 Project Overview

Traffic monitoring plays an important role in intelligent transportation systems. This project automatically classifies traffic images into predefined traffic density categories using a custom CNN architecture built from scratch.

The project demonstrates the complete deep learning workflow:

- Data preprocessing
- Data augmentation
- Custom CNN implementation
- Model training
- Validation
- Performance evaluation
- Single image prediction

---

## 🎯 Objectives

- Build a CNN from scratch using PyTorch.
- Classify traffic images into five categories.
- Handle class imbalance using Weighted CrossEntropyLoss.
- Evaluate performance using multiple metrics.
- Create an end-to-end prediction pipeline.

---

## 🧠 Classes

| Class ID | Traffic Category |
|----------|------------------|
| 0 | Empty |
| 1 | High |
| 2 | Low |
| 3 | Medium |
| 4 | Traffic Jam |

---

## 🏗 Project Structure

```text
traffic-image-classification/

├── Data/
│   ├── training/
│   ├── validation/
│   └── testing/
│
├── outputs/
│   ├── models/
│   ├── graphs/
│   ├── logs/
│   └── predictions/
│
├── src/
│   ├── config.py
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── README.md
├── LICENSE
└── .gitignore
```

---

## 🛠 Tech Stack

- Python
- PyTorch
- TorchVision
- NumPy
- PIL
- Matplotlib
- Scikit-learn

---

## 🧠 CNN Architecture

```
Input Image (128 × 128 × 3)

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

Flatten

↓

Fully Connected (64)

↓

Dropout

↓

Output Layer (5 Classes)
```

---

## 📊 Dataset

The dataset contains traffic images categorized into five traffic density classes.

### Dataset Split

- Training Images: **3378**
- Validation Images: **340**
- Testing Images: **320**

---

## ⚙ Training Configuration

| Parameter | Value |
|-----------|-------|
| Image Size | 128 × 128 |
| Batch Size | 8 |
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Loss Function | Weighted CrossEntropyLoss |
| Epochs | 20 |
| Device | CPU |

---

## 📈 Results

### Test Accuracy

**76.88%**

### Classification Performance

| Class | Precision | Recall | F1-Score |
|--------|----------:|--------:|----------:|
| Empty | 0.84 | 0.95 | 0.89 |
| High | 0.78 | 0.73 | 0.76 |
| Low | 0.72 | 0.53 | 0.61 |
| Medium | 0.63 | 0.67 | 0.65 |
| Traffic Jam | 0.85 | 0.95 | 0.90 |

---

## 🚀 Run Training

```bash
python src/train.py
```

---

## 📊 Evaluate Model

```bash
python src/evaluate.py
```

---

## 🔍 Predict a Single Image

```bash
python src/predict.py
```

---

## ✨ Features

- Custom CNN Architecture
- Data Augmentation
- Weighted Loss for Imbalanced Dataset
- Model Checkpoint Saving
- Evaluation Pipeline
- Single Image Prediction
- Training History Logging

---

## 📌 Future Improvements

- ResNet18 Transfer Learning
- MobileNetV3 Comparison
- Grad-CAM Visualization
- Streamlit Web Application
- Real-Time Traffic Classification
- Model Quantization for Edge Deployment

---

## 👨‍💻 Author

**Sathwik Chiliveri**

B.Tech Computer Science & Engineering

Machine Learning | Deep Learning | Computer Vision

GitHub: https://github.com/sathwikchiliveri

---

## ⭐ If you found this project useful

Give this repository a ⭐ on GitHub.

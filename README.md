# Digit Classification with Explainable AI (XAI)

A PyTorch-based CNN model for handwritten digit classification with integrated Explainability (Grad-CAM).

---

## Overview

This project implements a Convolutional Neural Network (CNN) for grayscale digit classification (28×28 images).
It follows a clean and modular design suitable for research, teaching, and reproducibility.
In addition to classification, it integrates explainable AI techniques (Grad-CAM) to visualize model reasoning and highlight the regions influencing predictions.

---

## Project Structure

Digit_Classification_XAI/
│
├── config/
│   └── config.yaml              # Hyperparameters and training configuration
│
├── data/
│   ├── train/                   # Training images organized by class
│   └── test/                    # Test images organized by class
│
├── src/
│   ├── main.py                  # Entry point for training and evaluation
│   ├── models/
│   │   └── cnn_model.py         # CNN model definition
│   ├── utils/
│   │   ├── dataset.py           # Data loading and augmentation
│   │   ├── train_eval.py        # Training and evaluation utilities
│   │   └── explain.py           # Grad-CAM and visualization methods
│
├── notebooks/
│   └── EDA_and_Explainability.ipynb   # Exploratory analysis and XAI demonstrations
│
├── checkpoints/                 # Saved model weights
│
├── README.md
└── requirements.txt

---

## Model Architecture

==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
CNNModel                                 [1, 10]                   --
├─Conv2d: 1-1                            [1, 16, 28, 28]           160
├─MaxPool2d: 1-2                         [1, 16, 14, 14]           --
├─Dropout: 1-3                           [1, 16, 14, 14]           --
├─Conv2d: 1-4                            [1, 32, 14, 14]           4,640
├─MaxPool2d: 1-5                         [1, 32, 7, 7]             --
├─Dropout: 1-6                           [1, 32, 7, 7]             --
├─Conv2d: 1-7                            [1, 64, 7, 7]             18,496
├─MaxPool2d: 1-8                         [1, 64, 3, 3]             --
├─Dropout: 1-9                           [1, 64, 3, 3]             --
├─Linear: 1-10                           [1, 20]                   11,540
├─Linear: 1-11                           [1, 10]                   210
==========================================================================================
Total params: 35,046
Trainable params: 35,046
Non-trainable params: 0
Total mult-adds (Units.MEGABYTES): 1.95
==========================================================================================
Input size (MB): 0.00
Forward/backward pass size (MB): 0.18
Params size (MB): 0.14
Estimated Total Size (MB): 0.32
==========================================================================================
---

## Training Summary

Model Evaluation Summary
-------------------------
Accuracy : 0.0960
Precision: 0.0213
Recall   : 0.0960
F1-Score : 0.0232

---

## Evaluation Results

**Confusion Matrix**

![Confusion Matrix](docs/confusion_matrix.png)

**Classification Report (Macro Averages)**

| Metric | Score |
|--------|--------|
| Accuracy | 0.956 |
| Precision | 0.957 |
| Recall | 0.956 |
| F1-score | 0.956 |

---

## Explainability (Grad-CAM)

Grad-CAM visualizations show where the CNN focuses when making predictions.
Misclassified samples are analyzed to understand model bias or confusion.

| True | Pred | Visualization |
|------|-------|---------------|
| 0 | 8 | ![GradCAM_0_8](docs/xai_0_8.png) |
| 2 | 6 | ![GradCAM_2_6](docs/xai_2_6.png) |
| 4 | 9 | ![GradCAM_4_9](docs/xai_4_9.png) |

---

## How to Run

### 1. Clone the repository
```bash
git clone https://github.com/<your_username>/Digit_Classification_XAI.git
cd Digit_Classification_XAI

### 2. Create environment
```bash
conda create -n pytorch_gpu python=3.10
conda activate pytorch_gpu
pip install -r requirements.txt

### 3. Train the model
```bash
python -m src.main

### 4. Explore explainability in Jupyter
```bash
jupyter notebook notebooks/EDA_and_Explainability.ipynb

Key Features:
- Modular PyTorch training pipeline
- Config-driven architecture
- Integrated Grad-CAM explainability
- Deterministic and reproducible results
- Compatible with CPU and GPU





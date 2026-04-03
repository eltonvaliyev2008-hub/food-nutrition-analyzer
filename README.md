# Food Nutrition Analyzer

Upload a food photo and instantly receive a complete nutritional breakdown — calories, protein, carbohydrates, fat, fiber, sugar, and sodium — adjusted per portion weight.

Live Demo: https://huggingface.co/spaces/EltonValiyev11/calorie-lens

---

## Overview

A deep learning–powered web application that classifies food images across 101 categories and returns per-gram nutritional data. Built with a ResNet101 transfer learning model trained on the Food-101 dataset, served via a FastAPI backend with a custom frontend.

---

## Pipeline

| Step | Description |
|------|-------------|
| Dataset | Food-101 — 101,000 images across 101 food categories (75,750 train / 25,250 validation) |
| Normalization | Dataset-specific mean [0.5459, 0.4444, 0.3444] and std [0.2625, 0.2659, 0.2706] computed from training set |
| Augmentation | RandomHorizontalFlip, RandomRotation(15°), RandomResizedCrop(scale=0.8–1.0), ColorJitter |
| Baseline | Custom FoodCNN — 3 convolutional blocks (3→32→64→128→256→512 channels), BatchNorm, MaxPool, Dropout(0.4) |
| Transfer Learning | ResNet101 pretrained on ImageNet — final FC layer replaced: 2048 → 101 |
| Training | Adam optimizer, StepLR scheduler, CrossEntropyLoss with label smoothing (0.1) |
| Deployment | FastAPI backend, custom HTML/CSS/JS frontend, hosted on Hugging Face Spaces via Docker |

---

## Model Comparison

| Metric | Custom CNN | ResNet101 |
|--------|------------|-----------|
| Input Resolution | 128×128 | 224×224 |
| Epochs | 30 | 20 |
| Learning Rate | 0.001 | 0.0001 |
| Scheduler | StepLR (step=7, gamma=0.5) | StepLR (step=10, gamma=0.5) |
| Deployed | No | Yes |

---

## ResNet101 — Training Configuration

```
Optimizer  : Adam (lr=0.0001, weight_decay=1e-4, betas=(0.9, 0.999))
Scheduler  : StepLR (step_size=10, gamma=0.5)
Loss       : CrossEntropyLoss (label_smoothing=0.1)
Batch size : 32
Epochs     : 20
Hardware   : GPU (CUDA)
```

---

## Application Features

- Top-3 predictions with confidence scores
- Nutritional breakdown across 7 metrics per selected portion weight (50g – 700g)
- Daily caloric intake percentage based on 2,000 kcal reference
- Drag-and-drop image upload with live preview
- Responsive design for both desktop and mobile

---

## Project Structure

```
food-nutrition-analyzer/
├── app.py                   # FastAPI application
├── templates/
│   └── index.html           # Frontend interface
├── yemek_qidaliliq/
│   └── nutrition.csv        # Per-food nutritional data
├── CNN.ipynb                # Custom CNN training notebook
├── resnet101.ipynb          # ResNet101 training notebook
├── requirements.txt
└── Dockerfile
```

---

## Tech Stack

- Python 3.10
- PyTorch 2.2 / TorchVision
- FastAPI + Uvicorn
- Hugging Face Hub (model storage)
- Docker (containerized deployment)

---

## Author

Elton Valiyev
https://www.linkedin.com/in/eltonvaliyev/

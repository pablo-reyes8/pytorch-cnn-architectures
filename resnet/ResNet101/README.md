# ResNet-101 Implementation

## Introduction

**ResNet (Residual Network)**, introduced by Kaiming He et al. in 2015, revolutionized deep learning by enabling the training of **very deep convolutional neural networks**. The key idea is the use of **residual (skip) connections**, which alleviate the vanishing gradient problem and allow effective optimization of architectures exceeding 100 layers.

This repository implements **ResNet-101 from scratch in PyTorch**, applied to the **Oxford-IIIT Pet dataset** (binary classification: cats vs dogs).  
Key highlights:

- Architecture based on **residual blocks** with identity and convolutional shortcuts.
- Modular design for **data loading, training, and evaluation**.
- Integration of **visualization utilities** for filters, activations, feature maps, and performance metrics.
- Configurable via **YAML file** for full reproducibility.

---

## Project Structure

The repository is organized into modular components:

### 1. `src/data/`

Data utilities.

- **`load_data.py`** – defines loaders for Oxford Pets dataset.
- **`utils_data.py`** – helper functions (mean/std computation, preprocessing).

### 2. `src/model/`

ResNet-101 model implementation.

- **`resnet_bloks.py`** – definition of residual blocks.
- **`restnet.py`** – main ResNet architecture (customizable depth).
- **`save_model.py`** – utility for saving trained models.

### 3. `src/training/`

Training pipeline.

- **`train_loop.py`** – main training loop.
- **`train_utils.py`** – optimization and logging helpers.

### 4. `src/testing_utils/`

Evaluation and visualization.

- **`evaluate_model.py`** – evaluation loop (loss, ROC-AUC, etc.).
- **`evaluate_plots.py`** – plots (ROC, confusion matrices, curves).
- **`test_utils.py`** – feature map visualizations, Grad-CAM, filters.

### 5. `test/`

Sanity check scripts.

- **`data_sanity_cheks.py`** – validates dataset integrity.
- **`model_sanity_cheks.py`** – tests model forward pass.

### 6. `experiments/`

Visual results generated during training/evaluation:

- Feature maps (per-layer visualization).
- Learned filters (Conv1, Conv2, Layer4).
- ROC Curve.
- Predictions on test set.

### 7. Config & Notebooks

- **`oxford_pets_binary_resnet101.yaml`** – experiment configuration (data, model, optimizer, scheduler, results).
- **`Resnet101.ipynb`** – Jupyter notebook with full training workflow.

---

## Training Setup

Training is configured via `oxford_pets_binary_resnet101.yaml`:

- **Dataset**: Oxford-IIIT Pet (binary task: cat vs dog).
- **Model**: ResNet-101-like (`blocks_per_stage=(3,4,23,3)`).
- **Device**: CUDA if available, else CPU.
- **Optimizer**: Adam (`lr=1e-3`, `betas=(0.9,0.999)`, `weight_decay=1e-4`).
- **Scheduler**: StepLR (`step_size=30, gamma=0.1`).
- **Epochs**: 30.
- **Normalization**: mean = `[0.4829, 0.4449, 0.3957]`, std = `[0.2592, 0.2532, 0.2598]`.

---

## Results

Final validation performance:

- **Val Loss**: 0.4084
- **ROC-AUC**: 0.9108

**Classification Report**:

| Class        | Precision | Recall | F1-Score   | Support |
| ------------ | --------- | ------ | ---------- | ------- |
| 0 (cat)      | 0.6840    | 0.8750 | 0.7678     | 240     |
| 1 (dog)      | 0.9301    | 0.8044 | 0.8627     | 496     |
| **Accuracy** |           |        | **0.8274** | 736     |

- **Macro Avg**: Precision 0.8071, Recall 0.8397, F1 0.8153
- **Weighted Avg**: Precision 0.8498, Recall 0.8274, F1 0.8318

---

## Educational Purpose

This project is designed for **learning and experimentation**:

- Implements **ResNet-101** from scratch for binary classification.
- Provides **data, training, testing, and visualization utilities**.
- Highlights the power of **residual connections** in deep architectures.
- Demonstrates reproducible workflows using YAML configs + Jupyter notebooks.

---

## References

- He, K., Zhang, X., Ren, S., & Sun, J. (2015). _Deep Residual Learning for Image Recognition_. CVPR.
- Oxford-IIIT Pet Dataset: [https://www.robots.ox.ac.uk/~vgg/data/pets/](https://www.robots.ox.ac.uk/~vgg/data/pets/)

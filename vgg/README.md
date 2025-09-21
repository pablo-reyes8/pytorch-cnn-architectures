# VGG16 From Scratch — Oxford-IIIT Pet (Binary & Multiclass)

## Introduction

This project implements **VGG16 with/without BatchNorm** **from scratch** in PyTorch and applies it to the **Oxford-IIIT Pet dataset** in two configurations:  
- **Multiclass (37 breeds)**  
- **Binary (cat vs. dog)**  

The focus is **educational and experimental**: weight initialization, dataset-specific normalization, data augmentation, warmup + cosine/step LR scheduling, gradient clipping, label smoothing, checkpoints, and evaluation/visualization utilities typical for CNNs (filters, feature maps, weight/activation histograms, confusion matrix, ROC curves, etc.).

---

## Project Structure

The repository is organized into modular components for clarity and reusability:

### 1. `src/data`
Dataset loading and preprocessing.  
- **`load_oxford_data.py`**: functions to create binary or multiclass loaders, compute dataset mean/std.  
- **`load_inat_data.py`**: optional utilities for alternative datasets.  
- **`view_data.py`**: quick visualization of samples.

### 2. `src/model`
Core VGG implementation and model utilities.  
- **`vgg_blocks.py`**: defines `VGGConvBlock` and `VGGDenseBlock`.  
- **`vgg16.py`**: complete VGG16 architecture (configurable for binary/multiclass, with/without BN).  
- **`save_model.py`**: save/load checkpoints (model, optimizer, scheduler).

### 3. `src/training`
Training utilities and loops.  
- **`train_loop.py`**: generic training pipeline supporting binary and multiclass modes.  

### 4. `src/testing`
Evaluation and experiment utilities.  
- **`evaluate_model.py`**: loss/metrics on validation or test sets, classification report, ROC/PR curves.  
- **`experiment_functions.py`**: visualization tools (filters, feature maps, weight/activation histograms, Grad-CAM optional).  

### 5. `tests`
Sanity checks and unit tests.  
- **`data_test.py`**: verifies dataset integrity, normalization, label mapping.  
- **`model_test.py`**: checks forward pass, output dimensions, and weight initialization.

### 6. `experiments`
Generated figures from trained models.  
- Feature maps, filters, histograms, ROC curves, etc.

### 7. Notebooks
Two complementary workflows:  
- **`vgg_training_showcase.ipynb`**: demonstrates training VGG16 with the modular utilities.  
- **`vgg_full.ipynb`**: a self-contained notebook containing the entire workflow.

---

## Educational Purpose

This project is designed for **learning and experimentation**. It:  
- Shows how to implement **VGG16 from scratch** with modular blocks.  
- Provides **visualization tools** for inspecting filters, feature maps, and learned weights.  
- Explores **binary vs. multiclass training** challenges on the Oxford-IIIT Pet dataset.  
- Highlights **training best practices** (weight init, normalization, LR scheduling, checkpointing).  

By replicating VGG16, this repository demonstrates why **deep convolutional backbones** remain a cornerstone in computer vision, and serves as a foundation for extending into modern architectures.

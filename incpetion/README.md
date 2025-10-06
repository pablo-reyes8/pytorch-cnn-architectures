# Inception v1 (GoogLeNet) — STL-10 (Multiclass)

## Introduction

This project implements **Inception v1 (GoogLeNet)** from scratch in PyTorch and applies it to the **STL-10 dataset** (10 classes, 96×96 images).  

The **motivation** behind Inception (Szegedy et al., *Going Deeper with Convolutions*, 2014) was to achieve **better representational efficiency**: instead of committing to a single receptive field, each Inception block computes **1×1, 3×3, 5×5 convolutions and pooling in parallel** and concatenates the results. This design allows the network to capture both fine and coarse features at the same stage.  

Key innovations include:  
- **1×1 convolutions** for dimensionality reduction before expensive filters.  
- **Parallel multi-scale feature extraction** within each block.  
- **Auxiliary classifiers (Aux1, Aux2)** to improve gradient flow and act as regularizers.  
- **Global average pooling** instead of large fully connected layers, reducing parameters.  

GoogLeNet won the **ILSVRC 2014 classification challenge**, showing that combining **depth with efficiency** could outperform very large models like VGG16 while using far fewer parameters (~6.8M vs. 138M).  

---

## Project Structure

The repository is organized into modular components:

### 1. `data/`
Dataset loading and preprocessing.  
- `load_data.py`: downloads and prepares STL-10, creates train/test splits, resizing images to 96×96.  
- `data_utils.py`: helper functions for normalization and augmentations.  

### 2. `model/`
Core Inception v1 implementation and supporting modules.  
- `inceptionv1.py`: complete GoogLeNet v1 with auxiliary heads.  
- `inception_aux_functions.py`: auxiliary classifiers (Aux1, Aux2).  
- `model_utils.py`: parameter counting, visualization hooks, forward checks.  
- `train_loop.py`: custom training loop with Inception-style loss.  

### 3. `training/`
Scripts for launching and monitoring training.  
- `Inception_full.ipynb`: main notebook demonstrating full training on STL-10.  

### 4. `testing/`
Sanity checks and evaluation utilities.  
- `sanity_checks.py`: forward/backward checks, gradient inspections, overfitting-on-one-batch test.  

### 5. `experiments/`
Figures generated during experiments:  
- Confusion matrix, ROC curve, Inception block activations, predictions at different stages.  

---

## Educational Purpose

This project is designed for **learning and experimentation**. It:  
- Explains the motivation and architecture of Inception v1.  
- Shows how to implement **Inception blocks and auxiliary classifiers** from scratch.  
- Provides **evaluation tools**: confusion matrix, ROC curves, sample predictions.  
- Demonstrates **training best practices**: AMP, gradient clipping, auxiliary losses.  

By replicating GoogLeNet v1, this repository illustrates how **architectural innovations** (multi-branch design, 1×1 convolutions, auxiliary heads) reshaped CNN design and paved the way for deeper and more efficient networks.


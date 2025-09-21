# ResNet-50 Implementation 

## Introduction

**ResNet (Residual Network)**, introduced by Kaiming He et al. in 2015, marked a major breakthrough in deep learning. ResNet won the **ILSVRC 2015 classification challenge**, achieving unprecedented accuracy while being dramatically deeper (up to 152 layers) than previous architectures.  

The key innovation of ResNet is the introduction of **residual connections (skip connections)**, which solved the problem of vanishing gradients in very deep networks. This design allowed training of networks with over 100 layers, something previously thought impractical.  

ResNet was revolutionary because it:  
- Demonstrated that **very deep networks** could be trained effectively using residual blocks.  
- Introduced the now-standard concept of **identity and convolutional shortcut connections**.  
- Provided a scalable architecture (ResNet-18, 34, 50, 101, 152) used as backbones for numerous tasks (detection, segmentation, transfer learning).  
- Became one of the most influential CNN architectures, forming the basis of modern vision models such as ResNeXt, EfficientNet, and even serving as encoders for vision transformers.

---

## Project Structure

This repository implements **ResNet-50** from scratch in PyTorch, applied to the CIFAR-10 dataset. The project is structured into data, model, training, and visualization utilities.

### 1. `load_data.py`
Dataset loading and preprocessing.  
- **`LoaderConfig`**: configuration object for dataset preprocessing and dataloaders.  
- **`create_cifar10_loaders(cfg: LoaderConfig)`**: prepares train/test loaders for CIFAR-10.  
- **`imshow_cifar`**: displays CIFAR-10 images for quick inspection.

### 2. `model.py`
Core ResNet-50 implementation.  
- **`IdentityBlock(nn.Module)`**: residual block without dimension changes (identity skip connection).  
- **`ConvolutionalBlock(nn.Module)`**: residual block with convolutional shortcut for dimension matching.  
- **`ResNet50(nn.Module)`**: complete ResNet-50 architecture built from stacked residual blocks.

### 3. `train_utils.py`
Training utilities for classification tasks.  
- **`_topk_accuracies`**: computes top-k accuracy metrics.  
- **`train_epoch_classification`**: trains the model for one epoch.  
- **`evaluate_classification`**: evaluates model performance on validation/test sets.

### 4. `test_utils.py`
Evaluation and visualization tools.  
- **`visualize_test_predictions`**: shows test predictions with optional error-only display.  
- **`show_feature_maps`**: visualizes intermediate activations from specific layers.  
- **`show_first_conv_filters`**: displays the learned filters of the first convolutional layer.  
- **`plot_weight_histograms`**: plots histograms of learned weights across layers.  
- **`plot_confusion_matrix`**: generates normalized confusion matrices for classification evaluation.  
- **`show_gradcam_grid`**: applies Grad-CAM to visualize class-specific saliency maps.

### 5. Jupyter Notebooks
As with other architectures in this collection, two complementary workflows are provided:  
- **`train_model.ipynb`**: demonstrates training ResNet-50 with the modular `.py` utilities.  
- **`full ResNet.ipynb`**: a single self-contained notebook containing the entire workflow.

---

## Educational Purpose

This project is intended for learning and experimentation. It:  
- Shows how to implement **ResNet-50** from scratch using residual blocks.  
- Provides visualization tools for inspecting filters, feature maps, and Grad-CAM saliency.  
- Uses CIFAR-10 as a manageable dataset for training and experimentation.  

By replicating ResNet-50, this repository highlights the significance of residual connections and why ResNet remains a **default backbone** in modern computer vision.

---

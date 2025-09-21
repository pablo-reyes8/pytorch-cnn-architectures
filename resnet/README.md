# ResNet Architectures Collection

## Introduction

This repository contains **from-scratch implementations of ResNet architectures in PyTorch**, designed for educational and experimental purposes.  
The goal is to understand how **residual connections** enable the training of very deep convolutional networks (ResNet family) and to provide a modular framework for data loading, training, evaluation, and visualization.

ResNet, introduced by **He et al. (2015)**, remains one of the most influential architectures in computer vision. It:

- Solved the vanishing gradient problem with **skip connections**.
- Enabled the training of networks deeper than 100 layers.
- Became a **default backbone** for classification, detection, segmentation, and transfer learning.

---

## Repository Structure

### 📂 `ResNet50/`

Implementation of **ResNet-50**, trained on **CIFAR-10**.

- Core files: `model.py`, `load_data.py`, `train_utils.py`, `test_utils.py`.
- Includes notebooks for training (`train_model.ipynb`) and full workflow (`ResNet50_full.ipynb`).
- Visualization tools: filters, feature maps, Grad-CAM, confusion matrices.
- Educational focus on replicating ResNet-50 as a standard deep CNN.

👉 See detailed [README inside ResNet50](./ResNet50/README.md).

---

### 📂 `ResNet101/`

Implementation of **ResNet-101**, trained on the **Oxford-IIIT Pet dataset** (binary classification: cats vs dogs).

- Modular pipeline (`src/data`, `src/model`, `src/training`, `src/testing_utils`).
- Experiment configuration via YAML (`oxford_pets_binary_resnet101.yaml`).
- Visual outputs stored in `experiments/` (filters, feature maps, ROC curve, predictions).
- Includes classification metrics (ROC-AUC = 0.91, Accuracy = 82.7%).

👉 See detailed [README inside ResNet101](./ResNet101/README.md).

---

## Educational Purpose

This repository is built for **learning and experimentation**, not for production.  
It demonstrates:

- How to implement deep CNN architectures **from scratch** in PyTorch.
- How to structure a project with **modular design** (data, model, training, evaluation).
- How to reproduce experiments with **YAML configs** and visualize results.
- Why **ResNet** remains a cornerstone in modern deep learning.

---

## References

- He, K., Zhang, X., Ren, S., & Sun, J. (2015). _Deep Residual Learning for Image Recognition_. CVPR.
- CIFAR-10 Dataset: [https://www.cs.toronto.edu/~kriz/cifar.html](https://www.cs.toronto.edu/~kriz/cifar.html)
- Oxford-IIIT Pet Dataset: [https://www.robots.ox.ac.uk/~vgg/data/pets/](https://www.robots.ox.ac.uk/~vgg/data/pets/)

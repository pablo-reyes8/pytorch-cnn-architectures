# EfficientNet v1 — Food-101 (Multiclass)

## 🧠 Introduction

This repository implements **EfficientNet v1** (*Tan & Le, 2019, “Rethinking Model Scaling for Convolutional Neural Networks”*) from scratch in **PyTorch**, applying it to the **Food-101** dataset (101 classes, ~100k images).

EfficientNet introduced the idea of **compound scaling** — instead of scaling only depth or width, the network scales *resolution*, *width* and *depth* **jointly and proportionally** using three coefficients:

$$
d = \alpha^\phi, \quad w = \beta^\phi, \quad r = \gamma^\phi
$$
subject to a constraint $\alpha \cdot \beta^2 \cdot \gamma^2 \approx 2$.

This simple principle allows EfficientNet-B0 to be expanded consistently up to B7, achieving **state-of-the-art accuracy/efficiency trade-offs** across multiple datasets.

Key contributions:
- 🔹 **Compound scaling** strategy with balanced growth of all dimensions.  
- 🔹 **MBConv** inverted bottlenecks with *depthwise separable convolutions*.  
- 🔹 **Squeeze-and-Excitation (SE)** attention for adaptive channel reweighting.  
- 🔹 **Stochastic Depth (Drop-Connect)** for better regularization in deep networks.  
- 🔹 A clean, parameter-efficient design achieving strong performance with fewer FLOPs.

EfficientNet-B0 achieves **~77% top-1 accuracy on ImageNet** with only **5.3 M parameters** — nearly 8× fewer than ResNet-152.


## 🗂️ Project Structure

The repository is organized into modular, fully testable components:

### 1. `data/`
Dataset loading and preprocessing.
- `load_data.py` — downloads and prepares **Food-101**, resizing images to 200×200.  
- `best_trainloaders.py` — optimized dataloaders with augmentation and caching.  
- `loaders_verification.py` — quick checks for dataset integrity and class balance.

### 2. `model/`
Core architecture and reusable building blocks.
- `cnn_utils.py` — convolutional helpers and padding logic (`ConvBNAct`).  
- `computer_scaler.py` — compound scaling utilities (`CompoundScaler`, `round_filters`, etc.).  
- `efficient_blocks.py` — **Squeeze-and-Excitation (SE)** and **Stochastic Depth** modules.  
- `MBConv.py` — implementation of the **MBConv** block with expand-depthwise-project stages.  
- `Efficient_Net.py` — full **EfficientNet v1** model (B0–B7) with compound scaling.  
- `train_loop.py` — AMP-ready training loop with gradient clipping and label smoothing.

### 3. `tests/`
Unit tests for every component.
- `test_utils_scaling.py` — checks scaling and divisibility logic.  
- `test_layers.py` — validates `ConvBNAct`, `SqueezeExcitation`, and `MBConv` behavior.  
- `test_efficientnet_shapes_and_params.py` — verifies forward passes and parameter growth across B0–B7.  
- `test_dynamic_resize_and_train_smoke.py` — smoke test for the `DynamicResize` layer and a mini training step.  
- `test_serialization.py` — ensures consistent save/load of model weights.  
- `test_param_breakdown.py` — parameter counting consistency tests.

### 4. `training/`
Scripts for launching experiments.
- `train_model.py` — complete training pipeline for Food-101.  
- `EfficientNet_full.ipynb` — notebook demonstrating compound scaling, stochastic depth, and results visualization.

## ⚙️ Technical Details

### Training Configuration
- **Optimizer:** RMSProp (`lr=0.064 × batch/256`, momentum=0.9, weight decay = 1e-5).  
- **Scheduler:** StepLR (decay = 0.96 every 8 epochs).  
- **Loss:** Cross-Entropy with optional `label_smoothing = 0.1`.  
- **Regularization:**  
  - Dropout (0.2 – 0.5 depending on φ)  
  - Stochastic Depth (`drop_connect_rate` ≈ 0.2 – 0.5)  
  - BatchNorm (ε = 1e-3, momentum = 0.99)  
- **Mixed Precision:** enabled via `torch.amp.autocast()` + `GradScaler`.  
- **Resolution scaling:** handled automatically by `DynamicResize` using γ^φ.

### Compound Scaling (α, β, γ)
| Model | φ | Depth (α^φ) | Width (β^φ) | Resolution (γ^φ) | Drop-Connect |
|:------|--:|-------------:|-------------:|------------------:|--------------:|
| B0 | 0 | 1.0 | 1.0 | 1.0 | 0.2 |
| B1 | 1 | 1.1 | 1.1 | 1.15 | 0.2 |
| B2 | 2 | 1.2 | 1.1 | 1.15² | 0.3 |
| B4 | 4 | 1.4 | 1.2 | 1.15⁴ | 0.4 |
| B7 | 7 | 2.0 | 1.4 | 1.15⁷ | 0.5 |

### Dataset
**Food-101**  
- 101 categories of dishes (~1000 images each).  
- Train = 75 750, Test = 25 250.  
- Augmentations: random crop 200×200 → resize γ^φ, horizontal flip, color jitter.  
- Normalization: ImageNet mean & std.

### Stochastic Depth
Each residual block receives an individual survival probability:
$$
p_i = 1 - \text{drop\_connect\_rate} \times \frac{i}{N}
$$
ensuring early blocks survive almost always while deeper ones are randomly skipped.

## 🎯 Educational Purpose

This project is built for **learning and experimentation**, not for production benchmarking.  
It demonstrates:

- How compound scaling couples **width, depth, and resolution**.  
- How to implement **MBConv + SE + Stochastic Depth** from scratch.  
- Proper **training loop design** with AMP, gradient clipping, and label smoothing.  
- How parameter efficiency can coexist with high representational power.

Each module is **unit-tested** in `tests/`, ensuring reliability and easier experimentation across model variants (B0–B7).

## 🧾 References

- Mingxing Tan & Quoc V. Le. *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.* ICML 2019.  
- Hu et al. *Squeeze-and-Excitation Networks.* CVPR 2018.  
- Sandler et al. *MobileNetV2: Inverted Residuals and Linear Bottlenecks.* CVPR 2018.

---

## ✍️ Author

Developed by **Pablo Reyes**  
*Economist | Data Scientist | ML Researcher*  
🔗 [github.com/pablo-reyes8](https://github.com/pablo-reyes8)

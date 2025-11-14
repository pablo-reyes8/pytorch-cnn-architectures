# Hybrid MobileNet Architectures

This repository contains a compact yet production-grade implementation of MobileNet models that can operate as either **v1** or **v2** from the very same class. Training tools, data loading utilities, lightweight differentiable augmentation, and a showcase notebook demonstrate how to adapt the architecture to Stanford Cars or any small/medium-scale visual task.

## Why This Project Matters
- **Unified API** – Instantiate MobileNetV1 or MobileNetV2 just by toggling a flag, while sharing blocks and layers where possible.
- **Configurable building blocks** – Depth-wise and point-wise convolutions expose dropout, activation, and channel width controls to quickly explore variants.
- **Reproducible training pipeline** – Modular one-epoch loops, early stopping, and differentiated augmentations provide reliable experimentation.
- **Professional scaffolding** – Pytest suites, documentation, requirements, and YAML configs make collaboration and deployment straightforward.

## Project Structure
```
mobilenet/
├── data/
│   └── load_data.py              # HuggingFace Stanford Cars dataloaders
├── model/
│   ├── depth_wise_block.py       # DepthwiseConv building block
│   ├── point_wise_block.py       # PointwiseConv building block
│   ├── mobiel_net_block.py       # Hybrid MobileNet block (v1/v2)
│   └── mobielnet.py              # MobileNet class with selectable version
├── testing/
│   ├── test_depthwise_block.py
│   ├── test_pointwise_block.py
│   ├── test_mobilenet_block.py
│   ├── test_mobilenet_model.py
│   └── test_training_utils.py    # Pytest suites for the full stack
├── training/
│   ├── diff_augment.py           # Differentiable augmentations
│   ├── one_epoch_loop.py         # Train/eval loops + accuracy helpers
│   └── training.py               # Early-stopping trainer wrapper
├── training_showcase/net_showcase.py.ipynb  # Notebook demo
├── mobilenet_models.yaml         # YAML with the showcased training setups
├── requirements.txt
└── README.md
```

## Installation
1. **Create a virtual environment (recommended)**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
3. (Optional) Install the [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/quick-start#login) if the Stanford Cars dataset requires authentication in your environment.

## Data Pipeline
`data/load_data.py` downloads `pkuHaowei/stanford-cars` from Hugging Face Datasets, filters the first `num_classes_limit` categories, and applies ImageNet-style normalization. Adjust:
- `num_classes_limit` to narrow/widen the label space.
- `img_size`, `batch_size`, `num_workers` to fit your hardware.

The loader returns `(train_loader, val_loader, train_dataset, val_dataset)` so you can quickly inspect batches in the notebook or scripts.

## Training the Model
```python
from model.mobielnet import MobileNet
from training.training import train_mobilenet, get_device
from training.diff_augment import DiffAugment
from data.load_data import get_stanford_cars_loaders

device = get_device()
train_loader, val_loader, *_ = get_stanford_cars_loaders(num_classes_limit=20)

model = MobileNet(num_classes=20, version="v2", width_mult=0.75, drop_out_rate=0.2)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
criterion = torch.nn.CrossEntropyLoss()
augment = DiffAugment(p_flip=0.5, p_contrast=0.5, p_cutout=0.5)

model, history = train_mobilenet(model, train_loader, val_loader,
                                 optimizer=optimizer,
                                 device=device,
                                 criterion=criterion,
                                 diff_augment=augment,
                                 num_epochs=50,
                                 patience=5,
                                 early_stop_delta=1e-3,
                                 acc_targets=(0.88, 0.94))
```
Key components:
- **`train_one_epoch` / `eval_one_epoch`** accumulate top-1/3/5 accuracies and return dict metrics.
- **`train_mobilenet`** adds early stopping either when validation loss plateaus or when target accuracies are met.
- **`DiffAugment`** keeps augmentations differentiable, so gradients stay intact even with flips, jitter, or cutout.

## Notebook Showcase
`training_showcase/net_showcase.py.ipynb` demonstrates the entire workflow:
1. Inspect filtered Stanford Cars batches.
2. Visualize augmentations and MobileNet blocks.
3. Train smaller v1/v2 variants to validate the pipeline.

Use it as a template for exploratory work or to reproduce plots/screenshots for reports.

## Model Configuration (YAML)
`mobilenet_models.yaml` mirrors the training scenarios from the notebook. Each entry describes:
- Architecture (version, width multiplier, custom block layout).
- Optimization hyper-parameters.
- Augmentation probabilities.
- Training & early-stopping criteria.

You can import the YAML into orchestrators or experiment trackers to keep runs consistent.

## Testing
Pytests live under `testing/` and cover low-level blocks, the hybrid MobileNet class, and training utilities.
```bash
python3 -m pytest testing
```
Add tests whenever you modify building blocks or trainers to keep the API stable.

## Extending the Repository
- Integrate new datasets by writing alternative loaders following the signature in `data/load_data.py`.
- Register additional MobileNet variants by passing custom `net1_layers` or `net2_layers` when instantiating `MobileNet`.
- Plug the YAML configs into Hydra, Lightning CLI, or your preferred experiment manager for large-scale sweeps.

## Acknowledgements
- MobileNetV1 & V2 papers by Howard et al. and Sandler et al.
- Stanford Cars dataset originally by Stanford AI Lab, served through the Hugging Face `pkuHaowei/stanford-cars` dataset card.

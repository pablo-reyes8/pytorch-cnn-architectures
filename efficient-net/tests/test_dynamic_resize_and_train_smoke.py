# tests/test_dynamic_resize_and_train_smoke.py
import torch
import torch.nn.functional as F
import pytest
from sklearn.metrics import f1_score

from model.efficent_blocks import * 
from model.compuder_scaler import * 
from model.Efficent_Net import *

def test_dynamic_resize_changes_spatial_size():
    resizer = DynamicResize()
    x = torch.randn(4, 3, 200, 200)
    y = resizer(x, 232)
    assert y.shape == (4, 3, 232, 232)

def test_smoke_one_training_step_no_nan():
    torch.manual_seed(0)
    device = torch.device("cpu")
    model = EfficientNet(num_classes=7, scaler=CompoundScaler(phi=0)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # batch sintético
    X = torch.randn(16, 3, 200, 200, device=device)
    y = torch.randint(0, 7, (16,), device=device)

    # “simula” B2 en resolución, manteniendo B0 en depth/width
    target_size = round_resolution(200, (1.15 ** 2), 8)
    resizer = DynamicResize()
    X = resizer(X, target_size)

    model.train()
    logits = model(X)
    loss = F.cross_entropy(logits, y, label_smoothing=0.1)
    assert torch.isfinite(loss)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    opt.step()
    opt.zero_grad()

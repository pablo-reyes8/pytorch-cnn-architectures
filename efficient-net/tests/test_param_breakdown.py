
import torch
import torch.nn as nn
import pytest

from model.Efficent_Net import * 
from model.compuder_scaler import *

def count_without_bn(model):
    total = 0
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm, nn.BatchNorm1d)):
            continue
        for p in m.parameters(recurse=False):
            total += p.numel()
    return total

@pytest.mark.slow
def test_param_counts_relative_to_phi():
    m0 = EfficientNet(num_classes=1000, scaler=CompoundScaler(phi=0))
    m2 = EfficientNet(num_classes=1000, scaler=CompoundScaler(phi=2))
    p0 = count_without_bn(m0)
    p2 = count_without_bn(m2)
    assert p2 > p0 * 1.5

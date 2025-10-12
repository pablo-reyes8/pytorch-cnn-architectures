
import torch
import pytest

from model.efficent_blocks import * 
from model. Efficent_Net import *

def count_params(m, trainable_only=False):
    params = [p for p in m.parameters() if (p.requires_grad or not trainable_only)]
    return sum(p.numel() for p in params)

@pytest.mark.parametrize("phi,input_size", [
    (0, 200),  # B0-like
    (2, 200),  # B2 depth/width (entrada 200 para probar GAP)
    (4, 224),  # B4 con otra entrada
])
def test_forward_shapes_any_input(phi, input_size):
    model = EfficientNet(num_classes=101, scaler=CompoundScaler(phi=phi))
    x = torch.randn(2, 3, input_size, input_size)
    y = model(x)
    assert y.shape == (2, 101)

def test_params_monotonic_increase_with_phi():
    m0 = EfficientNet(num_classes=101, scaler=CompoundScaler(phi=0))
    m2 = EfficientNet(num_classes=101, scaler=CompoundScaler(phi=2))
    m4 = EfficientNet(num_classes=101, scaler=CompoundScaler(phi=4))
    p0 = count_params(m0)
    p2 = count_params(m2)
    p4 = count_params(m4)
    assert p0 < p2 < p4  # más profundidad/ancho → más params

def test_drop_connect_schedule_increasing():
    # inspecciona los drop rates asignados bloque a bloque (requiere acceso a módulos MBConv)
    model = EfficientNet(num_classes=101, scaler=CompoundScaler(phi=0), drop_connect_rate=0.2)
    drop_rates = []
    for m in model.blocks.modules():
        if hasattr(m, "sd") and m.__class__.__name__ == "MBConv":
            # sd puede ser None si no hay residual
            if getattr(m, "sd", None) is not None:
                drop_rates.append(m.sd.p)
    # monotónico no estricto (algunos bloques sin residual se saltan)
    assert all(x <= y for x, y in zip(drop_rates, drop_rates[1:]))

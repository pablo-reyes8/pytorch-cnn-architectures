import torch

from model.mobielnet import MobileNet, count_params


def test_mobilenet_v1_forward_output_shape():
    model = MobileNet(num_classes=5, version="v1", width_mult=0.5)
    x = torch.randn(2, 3, 128, 128)
    with torch.no_grad():
        logits = model(x)
    assert logits.shape == (2, 5)


def test_mobilenet_v2_forward_output_shape():
    model = MobileNet(num_classes=3, version="v2", width_mult=1.0)
    x = torch.randn(1, 3, 160, 160)
    with torch.no_grad():
        logits = model(x)
    assert logits.shape == (1, 3)


def test_width_multiplier_reduces_parameter_count():
    large = MobileNet(num_classes=10, version="v2", width_mult=1.0)
    small = MobileNet(num_classes=10, version="v2", width_mult=0.5)
    assert count_params(small) < count_params(large)

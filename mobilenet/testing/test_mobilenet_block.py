import pytest
import torch

from model.mobiel_net_block import MobileNetBlock


def test_mobilenet_block_v1_downsamples_and_expands_channels():
    block = MobileNetBlock(in_channels=16, out_channels=32, stride=2, version="v1")
    x = torch.randn(1, 16, 32, 32)
    with torch.no_grad():
        y = block(x)
    assert y.shape == (1, 32, 16, 16)


def test_mobilenet_block_v2_residual_matches_block_plus_input():
    block = MobileNetBlock(in_channels=16, out_channels=16, stride=1, version="v2", expand_ratio=1)
    block.eval()
    x = torch.randn(1, 16, 32, 32)
    with torch.no_grad():
        block_only = block.block(x)
        y = block(x)
    assert torch.allclose(y, block_only + x, atol=1e-6)


def test_mobilenet_block_invalid_version_raises():
    with pytest.raises(AssertionError):
        MobileNetBlock(in_channels=8, out_channels=8, version="v3")

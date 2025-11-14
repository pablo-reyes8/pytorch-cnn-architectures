import torch

from model.depth_wise_block import DepthwiseConv


def test_depthwise_conv_preserves_channels_and_shape():
    layer = DepthwiseConv(in_channels=8, stride=1, padding=1)
    x = torch.randn(4, 8, 32, 32)
    with torch.no_grad():
        y = layer(x)
    assert y.shape == x.shape


def test_depthwise_conv_stride_reduces_spatial_resolution():
    layer = DepthwiseConv(in_channels=3, stride=2, padding=1)
    x = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        y = layer(x)
    assert y.shape[2:] == (32, 32)

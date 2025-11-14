import torch

from model.point_wise_block import PointwiseConv


def test_pointwise_conv_changes_channel_count():
    layer = PointwiseConv(in_channels=4, out_channels=7)
    x = torch.randn(2, 4, 16, 16)
    with torch.no_grad():
        y = layer(x)
    assert y.shape == (2, 7, 16, 16)


def test_pointwise_conv_without_activation_allows_negative_outputs():
    layer = PointwiseConv(in_channels=3, out_channels=3, act_layer=None)
    layer.eval()
    with torch.no_grad():
        layer.conv.weight.fill_(-1.0)
        layer.bn.weight.fill_(1.0)
        layer.bn.bias.zero_()

    x = torch.ones(1, 3, 8, 8)
    with torch.no_grad():
        y = layer(x)

    assert torch.all(y < 0), "Output should keep negative values when no activation is applied"

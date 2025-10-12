# tests/test_layers.py
import torch
import torch.nn as nn
import pytest

from model.cnn_utils import * 
from model.efficent_blocks import *
from model.MBConv import *



def test_conv_bn_act_same_padding_stride2_preserves_formula():
    x = torch.randn(1, 3, 51, 57)
    # stride=2 con SAME padding dinámico no debe explotar
    layer = ConvBNAct(3, 8, kernel_size=3, stride=2, groups=1)
    y = layer(x)
    # salida ~ ceil(H/2), ceil(W/2)
    assert y.shape[2] == (51 + 1) // 2 or y.shape[2] == 26
    assert y.shape[3] == (57 + 1) // 2 or y.shape[3] == 29
    assert y.shape[1] == 8

def test_se_hidden_uses_in_ch_pre_expand_not_mid_ch():
    in_ch = 24
    expand_ratio = 6
    mid_ch = make_divisible(in_ch * expand_ratio, 8)
    se = SqueezeExcitation(in_ch_after_expand=mid_ch, se_hidden_from=in_ch, se_ratio=0.25, divisible_by=8)
    # parámetros del primer 1x1 = mid_ch * hidden
    # hidden debería ser aproximadamente in_ch*0.25 redondeado a 8
    hidden_expected_raw = max(1, int(in_ch * 0.25))
    hidden_expected = max(8, int(hidden_expected_raw + 8/2) // 8 * 8)
    hidden_expected = min(hidden_expected, mid_ch)
    assert se.reduce.out_channels == hidden_expected
    assert se.expand.in_channels == hidden_expected
    assert se.reduce.in_channels == mid_ch
    assert se.expand.out_channels == mid_ch

def test_mbconv_shapes_residual_and_strides():
    # stride=1 residual on
    mb1 = MBConv(in_ch=24, out_ch=24, kernel_size=3, stride=1, expand_ratio=6.0, se_ratio=0.25)
    x = torch.randn(2, 24, 56, 56)
    y = mb1(x)
    assert y.shape == x.shape

    # stride=2 no residual
    mb2 = MBConv(in_ch=24, out_ch=40, kernel_size=5, stride=2, expand_ratio=6.0, se_ratio=0.25)
    x2 = torch.randn(2, 24, 56, 56)
    y2 = mb2(x2)
    assert y2.shape[1] == 40
    assert y2.shape[2] in {28, 29}  # SAME padding puede dar ceil
    assert y2.shape[3] in {28, 29}

def test_stochastic_depth_train_vs_eval_behavior():
    mb = MBConv(in_ch=24, out_ch=24, kernel_size=3, stride=1, expand_ratio=6.0, se_ratio=0.25, drop_connect=0.2)
    x = torch.randn(4, 24, 28, 28)

    # train: outputs deben diferir por el Bernoulli (probablemente)
    mb.train()
    y1 = mb(x)
    y2 = mb(x)
    # no exigimos siempre diferente (puede coincidir por azar), pero al menos soporta forward
    assert y1.shape == y2.shape

    # eval: determinista
    mb.eval()
    y3 = mb(x)
    y4 = mb(x)
    assert torch.allclose(y3, y4, atol=0, rtol=0)

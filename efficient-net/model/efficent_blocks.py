import torch.nn as nn
import torch
import math
from model.cnn_utils import *
import torch.nn.functional as F

class ConvBNAct(nn.Module):
    """
    Conv2d (+ ZeroPad2d para SAME con stride>1 opcional) + BatchNorm2d + SiLU
    Puede actuar como:
      - expand 1x1: kernel=1, stride=1, groups=1
      - depthwise: kernel=3/5, stride=1/2, groups=in_channels
    """
    def __init__(self, in_ch: int, out_ch: int,
        kernel_size: int = 1,
        stride: int = 1,
        groups: int = 1,
        use_act: bool = True,
        bn_eps: float = 1e-3,
        bn_mom: float = 0.99,):

        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = 1
        self.use_act = use_act
        self.groups = groups

        # Determina si necesita padding dinámico (solo stride > 1)
        self.use_dynamic_pad = stride > 1

        # same pading si stride = 1
        if stride == 1:
          pad = get_same_pad(kernel_size, stride, self.dilation)
        else:
          pad = 0

        self.pad_layer = None

        # Forward Pass bones
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size,stride=stride,
            padding=pad, dilation=self.dilation, groups=groups,bias=False)
        self.bn = nn.BatchNorm2d(out_ch, eps=bn_eps, momentum=bn_mom)

        if use_act:
          self.act = nn.SiLU(inplace=True)
        else:
          self.act = nn.Identity()

    def forward(self, x):
        if self.use_dynamic_pad:
            # calcular padding "same" en tiempo de ejecución
            l, r, t, b = calc_pad_2d(x, self.kernel_size, self.stride, self.dilation)
            if (l or r or t or b):
                x = F.pad(x, (l, r, t, b))

        # Forward
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class SqueezeExcitation(nn.Module):
    """
    Squeeze-and-Excitation (SE) block.
    GAP -> 1x1 reduce (C*r) -> SiLU -> 1x1 expand (C) -> Sigmoid -> channel reweight.
    EficientNet usa típicamente se_ratio=0.25 y redondeo divisible por 8.
    """

    def __init__(self, in_ch_after_expand: int,  # = mid_ch (entrada real al SE)
                 se_hidden_from: int,            # = in_ch (antes del expand)
                 se_ratio: float = 0.25,
                 divisible_by: int = 8):
        super().__init__()

        # canales ocultos (al menos 1, y divisible por 'divisible_by')
        hidden_raw = max(1, int(se_hidden_from * se_ratio))
        hidden = make_divisible(hidden_raw, divisible_by) if divisible_by > 0 else hidden_raw
        hidden = min(hidden, in_ch_after_expand)

        self.pool = nn.AdaptiveAvgPool2d(1)  # N,C,1,1

        # bottleneck (reduce -> SiLU -> expand)
        self.reduce = nn.Conv2d(in_ch_after_expand, hidden, 1)
        self.act = nn.SiLU(inplace=True)
        self.expand = nn.Conv2d(hidden, in_ch_after_expand, 1)

        self.gate = nn.Sigmoid()

    def forward(self, x: torch.Tensor):
        s = self.pool(x)
        s = self.reduce(s)
        s = self.act(s)
        s = self.expand(s)
        s = self.gate(s)
        return x * s  # reescala canal-a-canal


class StochasticDepth(nn.Module):
    """
    Stochastic Depth / Drop-Connect del ramo principal (main path).
    En entrenamiento, con prob p se "apaga" el bloque (x) y se deja pasar el skip (residual).
    En inferencia, se desactiva (equivale a x + residual).
    """
    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = float(p)


    def forward(self, x, residual):
        if (not self.training) or self.p == 0.0:
            return x + residual

        keep = 1.0 - self.p
        # máscara por muestra (N,1,1,1)
        mask = torch.empty((x.shape[0], 1, 1, 1), device=x.device, dtype=x.dtype).bernoulli_(keep)
        x = x * (mask / keep)
        return x + residual
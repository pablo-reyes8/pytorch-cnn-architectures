import torch.nn as nn
import torch
import math
from model.cnn_utils import *
import torch.nn.functional as F

from model.efficent_blocks import *

class MBConv(nn.Module):
    """
    MBConv (EfficientNet v1):
      x -> [Expand 1x1 (t)] -> [Depthwise kxk] -> [SE] -> [Project 1x1 (linear)]
      + residual si stride==1 y C_in==C_out (con stochastic depth opcional).

    Args:
        in_ch: canales de entrada
        out_ch: canales de salida
        kernel_size: {3,5,7}
        stride: {1,2}
        expand_ratio: puede ser int o float (t); si ==1 no hay expand
        se_ratio: típicamente 0.25 en EfficientNet
        drop_connect: prob. de stochastic depth para ESTE bloque (programada fuera)
        bn_eps, bn_mom: BatchNorm estilo TF (1e-3, 0.99)
        divisible_by: redondeo de canales (8 por defecto, estilo EfficientNet/MobileNet)
    """

    def __init__(self,in_ch: int, out_ch: int, kernel_size: int = 3,stride: int = 1,
        expand_ratio: int = 6,
        se_ratio: float = 0.25,
        drop_connect: float = 0.0,
        bn_eps: float = 1e-3,
        bn_mom: float = 0.99 , divisible_by: int = 8):

        super().__init__()
        assert stride in (1, 2), "stride debe ser 1 o 2"
        assert kernel_size in (3, 5, 7) , "kernel_size debe ser 3, 5 o 7"


        self.use_residual = (stride == 1 and in_ch == out_ch)

        # canales intermedios (expand)
        t = float(expand_ratio)
        if t <= 1.0 + 1e-8:
            mid_ch = in_ch
            self.expand = nn.Identity()
        else:
            mid_ch = make_divisible(in_ch * t, divisible_by)
            self.expand = ConvBNAct(
                in_ch=in_ch, out_ch=mid_ch,
                kernel_size=1, stride=1, groups=1,
                use_act=True, bn_eps=bn_eps, bn_mom=bn_mom)

        # Depthwise
        self.depthwise = ConvBNAct(mid_ch, mid_ch, kernel_size=kernel_size, stride=stride, groups=mid_ch,
            use_act=True, bn_eps=bn_eps, bn_mom=bn_mom)

        # SE
        self.se = SqueezeExcitation(in_ch_after_expand=mid_ch, se_hidden_from=in_ch,
            se_ratio=se_ratio,divisible_by=divisible_by)

        # Project (1x1, sin activación)
        self.project = ConvBNAct(mid_ch, out_ch, kernel_size=1, stride=1, groups=1,
            use_act=False, bn_eps=bn_eps, bn_mom=bn_mom)

        # Stochastic depth
        if self.use_residual and drop_connect > 0.0:
          self.sd = StochasticDepth(drop_connect)
        else:
          self.sd = None

    def forward(self, x: torch.Tensor):
        identity = x if self.use_residual else None

        x = self.expand(x)
        x = self.depthwise(x)
        x = self.se(x)
        x = self.project(x)

        if self.use_residual:
            if self.sd is not None:
                x = self.sd(x, identity)
            else:
                x = x + identity

        return x
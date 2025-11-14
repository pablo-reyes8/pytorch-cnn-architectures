import torch
import torch.nn as nn

from model.depth_wise_block import * 
from model.point_wise_block import *

class MobileNetBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        version="v1",
        expand_ratio=6, dropout_rate=0.0  #típico de MobileNet v2
    ):
        super().__init__()

        assert version in ("v1", "v2"), "version debe ser 'v1' o 'v2'"
        self.version = version

        if version == "v1":
            act = nn.ReLU
        else:
            act = nn.ReLU6

        if version == "v1":
            self.use_residual = False

            self.block = nn.Sequential(
                DepthwiseConv(in_channels=in_channels,
                    kernel_size=3,stride=stride,
                    padding=1,bias=False,act_layer=act, dropout_rate=dropout_rate),
                
                PointwiseConv(in_channels=in_channels,
                    out_channels=out_channels,bias=False,act_layer=act, dropout_rate=dropout_rate))

        else:  # version == "v2"
            #  Ej 96x96x32 (proyeccion) -> 96x96x192 (aplicamos conv kernel=3) -> 96x96x32 (proyeccion)

            hidden_dim = int(round(in_channels * expand_ratio)) # Expandimos los canales para V2 trick
            # Res skip solo si mismo num de canales y stride=1
            self.use_residual = (stride == 1 and in_channels == out_channels)

            layers = []

            if expand_ratio != 1:
                layers.append(
                    PointwiseConv(in_channels=in_channels,
                        out_channels=hidden_dim, bias=False, act_layer=act, dropout_rate=dropout_rate)) # De por si 1x1 conv2d
            else:
                hidden_dim = in_channels

            layers.append(
                DepthwiseConv(in_channels=hidden_dim,
                    kernel_size=3,
                    stride=stride, padding=1,bias=False,act_layer=act, dropout_rate=dropout_rate)) # El valor de salida es el mismo entrada por eso no hay out 
            
            layers.append(
                PointwiseConv(
                    in_channels=hidden_dim,
                    out_channels=out_channels,
                    bias=False,
                    act_layer=None, dropout_rate=dropout_rate)) # Volvemos a la resolucucion original (bottelneck)

            self.block = nn.Sequential(*layers)

    def forward(self, x):
        out = self.block(x)

        if self.version == "v2" and self.use_residual:
            out = out + x # Skip conections

        return out
import torch
import torch.nn as nn

class DepthwiseConv(nn.Module):
    def __init__(
        self,
        in_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        act_layer=nn.ReLU,
        dropout_rate: float = 0.0,  
    ):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=bias,
        )

        self.bn = nn.BatchNorm2d(in_channels)

        # Dropout opcional
        if dropout_rate > 0:
            self.dropout = nn.Dropout2d(dropout_rate)
        else:
            self.dropout = nn.Identity()

        # Activación
        if act_layer is not None:
            self.act = act_layer(inplace=True)
        else:
            self.act = nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.dropout(x)  
        x = self.act(x)
        return x
    
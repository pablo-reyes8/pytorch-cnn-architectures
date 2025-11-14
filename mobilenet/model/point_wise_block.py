import torch
import torch.nn as nn

class PointwiseConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        bias=False,
        act_layer=nn.ReLU,
        dropout_rate: float = 0.0,   
    ):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )

        self.bn = nn.BatchNorm2d(out_channels)

        # Dropout opcional
        if dropout_rate > 0:
            self.dropout = nn.Dropout2d(dropout_rate)
        else:
            self.dropout = nn.Identity()

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
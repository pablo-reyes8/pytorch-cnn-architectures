import torch.nn as nn
import torch


def init_vgg_weights(module: nn.Module):
    """
    Aplica la inicialización recomendada para VGG:
      - Conv2d: Kaiming normal (fan_out), bias=0
      - Linear: Kaiming normal (fan_in), bias=0
      - BatchNorm2d: weight=1, bias=0
    """
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)

class VGGConvBlock(nn.Module):
    """
    Bloque típico VGG:
      [Conv3x3 -> (BN) -> ReLU] x num_convs  +  (MaxPool 2x2 opcional)

    Args:
        in_channels (int): canales de entrada al bloque.
        out_channels (int): canales de salida de CADA conv del bloque (constantes dentro del bloque).
        num_convs (int): cuántas convs 3x3 apiladas (p.ej., 2 o 3 como en VGG-16/19).
        use_bn (bool): si True, inserta BatchNorm después de cada Conv2d.
        pool (bool): si True, añade MaxPool2d(kernel=2, stride=2) al final.
    """

    def __init__(self, in_channels,
        out_channels, num_convs,*, use_bn = False, pool = True):

        super().__init__()
        layers = []
        c_in = in_channels

        for _ in range(num_convs):
            layers.append(nn.Conv2d(c_in, out_channels, kernel_size=3, padding=1, bias=not use_bn))

            if use_bn:
                layers.append(nn.BatchNorm2d(out_channels))

            layers.append(nn.ReLU(inplace=True))
            c_in = out_channels

        if pool:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        self.block = nn.Sequential(*layers)
        self.apply(init_vgg_weights)

    def forward(self, x: torch.Tensor):
        return self.block(x)


class VGGDenseBlock(nn.Module):
    """
    Bloque denso típico de VGG:
      Linear -> ReLU -> Dropout

    Args:
        in_features (int)
        out_features (int)
        dropout (float): prob. de Dropout (0.5 clásico en VGG). Si 0.0, no se añade Dropout.
    """

    def __init__(self,in_features,out_features,*,dropout = 0.5):

        super().__init__()
        layers = [
            nn.Linear(in_features, out_features, bias=True),
            nn.ReLU(inplace=True),]

        if dropout and dropout > 0.0:
            layers.append(nn.Dropout(p=dropout))

        self.block = nn.Sequential(*layers)
        self.apply(init_vgg_weights)

    def forward(self, x: torch.Tensor):
        return self.block(x)
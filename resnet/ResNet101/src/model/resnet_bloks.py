import torch
import torch.nn as nn


class IdentityBlock(nn.Module):
    """
    ResNet v1 'identity block' (bottleneck) en PyTorch.
    Estructura: 1x1 -> 3x3 -> 1x1 con BN + ReLU (post-activation después de la suma).

    Args:
        in_channels (int): canales de entrada (debe ser igual a F3 en 'filters').
        f (int): tamaño del kernel para la conv central (3 en el paper clásico).
        filters (tuple[int, int, int]): (F1, F2, F3) # filtros en 1x1, 3x3, 1x1.
        init (str): 'uniform' (similar a random_uniform) o 'kaiming' (recomendado).
        init_range (float): rango para init uniform ([-init_range, init_range]).
        bn_eps (float): eps de BatchNorm2d.
        bn_momentum (float): momentum de BatchNorm2d.

    Notas:
        - Este bloque NO cambia dimensiones; requiere in_channels == F3.
        - Post-activation: ReLU después de la suma (como en ResNet v1 original).
    """

    def __init__(self,in_channels,f,
        filters,
        init = "uniform",
        init_range = 0.05,
        bn_eps = 1e-5,
        bn_momentum = 0.1,):

        super().__init__()
        F1, F2, F3 = filters

        assert in_channels == F3, ("IdentityBlock requiere que in_channels == F3 (no hay proyección en el atajo). "
            f"Recibido in={in_channels}, F3={F3}")

        # Conv-BN-ReLU (1): 1x1 reduce canales F3 -> F1
        self.conv1 = nn.Conv2d(in_channels, F1, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1   = nn.BatchNorm2d(F1, eps=bn_eps, momentum=bn_momentum)

        # Conv-BN-ReLU (2): fxf con 'same' padding (f//2) F1 -> F2
        self.conv2 = nn.Conv2d(F1, F2, kernel_size=f, stride=1, padding=f//2, bias=False)
        self.bn2   = nn.BatchNorm2d(F2, eps=bn_eps, momentum=bn_momentum)

        # Conv-BN (3): 1x1 expande F2 -> F3
        self.conv3 = nn.Conv2d(F2, F3, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3   = nn.BatchNorm2d(F3, eps=bn_eps, momentum=bn_momentum)

        self.relu = nn.ReLU(inplace=True)

        # Inicialización (equivalente a random_uniform o kaiming para ResNets)
        self._init_weights(init=init, init_range=init_range)

    def _init_weights(self, init: str = "uniform", init_range: float = 0.05):
        modules = [self.conv1, self.conv2, self.conv3]
        if init == "uniform":
            for m in modules:
                nn.init.uniform_(m.weight, -init_range, init_range)
        elif init == "kaiming":
            for m in modules:
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        else:
            raise ValueError(f"init '{init}' no soportada. Usa 'uniform' o 'kaiming'.")
        
        for bn in [self.bn1, self.bn2, self.bn3]:
            nn.init.ones_(bn.weight)
            nn.init.zeros_(bn.bias)

    def forward(self, x: torch.Tensor):
        shortcut = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out = out + shortcut
        out = self.relu(out)
        return out


class ConvolutionalBlock(nn.Module):
    """
    ResNet v1 'convolutional block' (bottleneck con proyección en el atajo).
    Estructura (main path): 1x1 (stride s) -> 3x3 (stride 1) -> 1x1 (stride 1)
    Atajo: 1x1 (stride s) para igualar dimensiones (espaciales y canales).

    Args:
        in_channels (int): canales de entrada.
        f (int): tamaño del kernel para la conv central (típicamente 3).
        filters (tuple[int, int, int]): (F1, F2, F3) para 1x1 -> 3x3 -> 1x1.
        s (int): stride para el downsample (en la 1x1 del main path y el atajo).
        init (str): 'uniform' (similar a glorot_uniform/random_uniform) o 'kaiming'.
        init_range (float): rango para init uniform ([-init_range, init_range]).
        bn_eps (float): eps de BatchNorm2d.
        bn_momentum (float): momentum de BatchNorm2d.
    """
    def __init__(self,in_channels , f, filters, s = 2,
        init= "uniform",init_range = 0.05,
        bn_eps = 1e-5,bn_momentum = 0.1,):

        super().__init__()
        F1, F2, F3 = filters

        # MAIN PATH
        self.conv1 = nn.Conv2d(in_channels, F1, kernel_size=1, stride=s, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(F1, eps=bn_eps, momentum=bn_momentum)

        self.conv2 = nn.Conv2d(F1, F2, kernel_size=f, stride=1, padding=f//2, bias=False)
        self.bn2 = nn.BatchNorm2d(F2, eps=bn_eps, momentum=bn_momentum)

        self.conv3 = nn.Conv2d(F2, F3, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3= nn.BatchNorm2d(F3, eps=bn_eps, momentum=bn_momentum)


        # SHORTCUT PATH (proyección 1x1 con stride s)
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, F3, kernel_size=1, stride=s, padding=0, bias=False),
            nn.BatchNorm2d(F3, eps=bn_eps, momentum=bn_momentum),)

        self.relu = nn.ReLU(inplace=True)
        self._init_weights(init=init, init_range=init_range)


    def _init_weights(self, init: str = "uniform", init_range: float = 0.05):
        convs = [self.conv1, self.conv2, self.conv3, self.shortcut[0]]
        if init == "uniform":
            for m in convs:
                nn.init.uniform_(m.weight, -init_range, init_range)
        elif init == "kaiming":
            for m in convs:
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        else:
            raise ValueError(f"init '{init}' no soportada. Usa 'uniform' o 'kaiming'.")
        for bn in [self.bn1, self.bn2, self.bn3, self.shortcut[1]]:
            nn.init.ones_(bn.weight)
            nn.init.zeros_(bn.bias)

    def forward(self, x: torch.Tensor):
        # Shortcut proyectado
        sc = self.shortcut(x)

        # Main path
        out = self.conv1(x); out = self.bn1(out); out = self.relu(out)
        out = self.conv2(out); out = self.bn2(out); out = self.relu(out)
        out = self.conv3(out); out = self.bn3(out)

        # Suma + ReLU (post-activation)
        out = self.relu(out + sc)
        return out
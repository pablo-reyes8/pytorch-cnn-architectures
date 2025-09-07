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

    def __init__(
        self,
        in_channels: int,
        f: int,
        filters: tuple,
        init: str = "uniform",
        init_range: float = 0.05,
        bn_eps: float = 1e-5,
        bn_momentum: float = 0.1,):

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
        # BN: gamma=1, beta=0
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
    def __init__(
        self,
        in_channels: int,
        f: int,
        filters: tuple,
        s: int = 2,
        init: str = "uniform",
        init_range: float = 0.05,
        bn_eps: float = 1e-5,
        bn_momentum: float = 0.1,):


        super().__init__()
        F1, F2, F3 = filters

        # MAIN PATH
        self.conv1 = nn.Conv2d(in_channels, F1, kernel_size=1, stride=s, padding=0, bias=False)
        self.bn1   = nn.BatchNorm2d(F1, eps=bn_eps, momentum=bn_momentum)

        self.conv2 = nn.Conv2d(F1, F2, kernel_size=f, stride=1, padding=f//2, bias=False)
        self.bn2   = nn.BatchNorm2d(F2, eps=bn_eps, momentum=bn_momentum)

        self.conv3 = nn.Conv2d(F2, F3, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3   = nn.BatchNorm2d(F3, eps=bn_eps, momentum=bn_momentum)

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
        # BN: gamma=1, beta=0
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


class ResNet50(nn.Module):
    """
    ResNet-50 v1 (post-activation) con selección del tipo de primer bloque de cada stage:
    - first_block: 'conv'  -> ConvolutionalBlock (proyección 1x1, permite stride>1)
    - first_block: 'identity' -> IdentityBlock (sin proyección, requiere mismas dims)

    Stages:
        conv1  -> 7x7 s=2, maxpool 3x3 s=2
        conv2_x: [3] blocks  -> (F1,F2,F3) = (64, 64, 256), stride 1 en primer bloque si 'conv'
        conv3_x: [4] blocks  -> (128,128,512),  stride 2 en primer bloque si 'conv'
        conv4_x: [6] blocks  -> (256,256,1024), stride 2 en primer bloque si 'conv'
        conv5_x: [3] blocks  -> (512,512,2048), stride 2 en primer bloque si 'conv'
    """

    def __init__(
        self,
        num_classes: int = 1000,
        first_block: str = "conv",   
        init: str = "kaiming",      
        bn_eps: float = 1e-5,
        bn_momentum: float = 0.1,):

        super().__init__()
        assert first_block in {"conv", "identity"}

        self.first_block = first_block
        self.init = init
        self.bn_eps = bn_eps
        self.bn_momentum = bn_momentum

        # ----- STEM (ImageNet-style) -----
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1   = nn.BatchNorm2d(64, eps=bn_eps, momentum=bn_momentum)
        self.relu  = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        in_ch = 64  # canales que salen del stem

        # ----- STAGES -----
        # conv2_x: 3 bloques, (64,64,256), stride para primer bloque: 1
        self.layer2, in_ch = self._make_stage(
            in_channels=in_ch,
            filters=(64, 64, 256),
            blocks=3,
            first_stride=1)

        # conv3_x: 4 bloques, (128,128,512), stride para primer bloque: 2
        self.layer3, in_ch = self._make_stage(
            in_channels=in_ch, # 256 
            filters=(128, 128, 512),
            blocks=4,
            first_stride=2,)

        # conv4_x: 6 bloques, (256,256,1024), stride para primer bloque: 2
        self.layer4, in_ch = self._make_stage(
            in_channels=in_ch, # 512 
            filters=(256, 256, 1024),
            blocks=6,
            first_stride=2,)

        # conv5_x: 3 bloques, (512,512,2048), stride para primer bloque: 2
        self.layer5, in_ch = self._make_stage(
            in_channels=in_ch, # 1024
            filters=(512, 512, 2048),
            blocks=3,
            first_stride=2,)

        # ----- HEAD -----
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(2048, num_classes)

        self._init_stem()


    def _init_stem(self):
        # Inicializa el stem (el resto se inicializa dentro de los bloques)
        if self.init == "kaiming":
            nn.init.kaiming_normal_(self.conv1.weight, mode="fan_out", nonlinearity="relu")
        else:
            nn.init.uniform_(self.conv1.weight, -0.05, 0.05)
        nn.init.ones_(self.bn1.weight)
        nn.init.zeros_(self.bn1.bias)


    def _make_stage(self,
        in_channels: int,
        filters: tuple,
        blocks: int,
        first_stride: int,):
        """
        Crea un stage: primer bloque puede ser 'conv' o 'identity' (según self.first_block),
        seguido de (blocks-1) identity blocks.
        """

        layers = []
        F1, F2, F3 = filters

        # Primer bloque del stage
        if self.first_block == "conv":

            layers.append(ConvolutionalBlock(in_channels=in_channels,
                    f=3,
                    filters=(F1, F2, F3),
                    s=first_stride,
                    init=self.init,
                    bn_eps=self.bn_eps,
                    bn_momentum=self.bn_momentum,))
        else:

            if not (first_stride == 1 and F3 == in_channels):
                raise ValueError(
                    "first_block='identity' requiere first_stride=1 y F3==in_channels "
                    f"(got stride={first_stride}, F3={F3}, in={in_channels}).")
            
            layers.append(IdentityBlock(in_channels=in_channels,
                    f=3,
                    filters=(F1, F2, F3),
                    init=self.init,
                    bn_eps=self.bn_eps,
                    bn_momentum=self.bn_momentum,))
            
        # Bloques restantes: siempre identity (mantienen dims dentro del stage)
        for _ in range(blocks - 1):
            layers.append(IdentityBlock(in_channels=F3,
                    f=3,
                    filters=(F1, F2, F3),
                    init=self.init,
                    bn_eps=self.bn_eps,
                    bn_momentum=self.bn_momentum,))

        return nn.Sequential(*layers), F3 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Stem
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        # Stages
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        # Head
        x = self.avgpool(x).flatten(1)
        x = self.fc(x)
        return x
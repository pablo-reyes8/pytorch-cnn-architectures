
import torch 
import torch.nn as nn 
from src.model.resnet_bloks import *

class ResNet(nn.Module):
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

    def __init__(self,
        num_classes: int = 1000,
        first_block: str = "conv",
        init: str = "kaiming",
        bn_eps: float = 1e-5,
        bn_momentum: float = 0.1,
        blocks_per_stage=(3,4,6,3)):

        super().__init__()
        self.first_block = first_block
        self.init = init
        self.bn_eps = bn_eps
        self.bn_momentum = bn_momentum

        # Stem = Capa inicial
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64, eps=bn_eps, momentum=bn_momentum)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)


        in_ch = 64

        # Stages
        cfgs = [(64, 64, 256, 1),
                 (128,128,512, 2),
                 (256,256,1024,2),
                 (512,512,2048,2)]

        layers = []
        for (F1, F2, F3, first_stride), n_blocks in zip(cfgs, blocks_per_stage):
            layer, in_ch = self._make_stage(in_ch, (F1,F2,F3), n_blocks, first_stride)
            layers.append(layer)

        self.layer2, self.layer3, self.layer4, self.layer5 = layers

        # Head
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(2048, num_classes)
        self._init_stem()


    def _make_stage(self,
        in_channels: int,
        filters: tuple,
        blocks: int,
        first_stride: int):

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

        # Bloques restantes: siempre identity 
        for _ in range(blocks - 1):
            layers.append(IdentityBlock(in_channels=F3,
                    f=3, filters=(F1, F2, F3),
                    init=self.init,
                    bn_eps=self.bn_eps,
                    bn_momentum=self.bn_momentum,))

        return nn.Sequential(*layers), F3
    
    def _init_stem(self):
        if self.init == "kaiming":
            nn.init.kaiming_normal_(self.conv1.weight, mode="fan_out", nonlinearity="relu")
        else:
            nn.init.uniform_(self.conv1.weight, -0.05, 0.05)
        nn.init.ones_(self.bn1.weight)
        nn.init.zeros_(self.bn1.bias)


    def forward(self, x: torch.Tensor):
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
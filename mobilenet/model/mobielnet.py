import torch
import torch.nn as nn

from model.mobiel_net_block import *

def count_params(model):
    return sum(p.numel() for p in model.parameters())


class MobileNet(nn.Module):
    def __init__(
        self,
        num_classes=1000, #ImageNet
        version="v1",
        width_mult=1.0 , drop_out_rate=0.2 , 
        net1_layers = [
                (64,   1),
                (128,  2),
                (128,  1),
                (256,  2),
                (256,  1),
                (512,  2),
                (512,  1),
                (512,  1),
                (512,  1),
                (512,  1),
                (512,  1),
                (1024, 2),
                (1024, 1)] , 
        net2_layers = [
                # t,  c,   n,  s
                [1,  16,  1,  1],
                [6,  24,  2,  2],
                [6,  32,  3,  2],
                [6,  64,  4,  2],
                [6,  96,  3,  1],
                [6, 160,  3,  2],
                [6, 320,  1,  1]]):
      
        super().__init__()
        assert version in ("v1", "v2"), "version debe ser 'v1' o 'v2'"

        self.version = version
        self.width_mult = width_mult

        def _make_divisible(v, divisor=8):
            # Función típica de MobileNetV2 para redondear canales
            return max(divisor, int(v + divisor / 2) // divisor * divisor)

        # STEM: primera conv
        if version == "v1":
            input_channel = int(32 * width_mult)
            self.stem = nn.Sequential(
                nn.Conv2d(3, input_channel, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(input_channel),
                nn.ReLU(inplace=True),)

            # Config exacta de MobileNetV1:
            # lista de (out_channels_base, stride)
            cfg = net1_layers

            layers = []
            in_c = input_channel
            for out_c_base, stride in cfg:

                out_c = int(out_c_base * width_mult)

                layers.append(
                    MobileNetBlock(
                        in_channels=in_c,
                        out_channels=out_c,
                        stride=stride,
                        version="v1", dropout_rate=drop_out_rate))
                
                in_c = out_c

            self.features = nn.Sequential(*layers) # Features Mobielnetv1
            self.last_channel = in_c  # canales del último bloque

            # MLP para clasificacion
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.classifier = nn.Linear(self.last_channel, num_classes)

        else:
            # MobileNetV2
            input_channel = _make_divisible(32 * width_mult, 8)

            self.stem = nn.Sequential(
                nn.Conv2d(3, input_channel, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(input_channel),
                nn.ReLU6(inplace=True))

            # Configuración MobileNetV2 (t, c, n, s)
            inverted_residual_settings = net2_layers

            layers = []
            in_c = input_channel

            for t, c, n, s in inverted_residual_settings:
                out_c = _make_divisible(c * width_mult, 8)

                for i in range(n):
                    stride = s if i == 0 else 1
                    layers.append(
                        MobileNetBlock(
                            in_channels=in_c,
                            out_channels=out_c,
                            stride=stride,
                            version="v2",
                            expand_ratio=t, dropout_rate=drop_out_rate))
                    
                    in_c = out_c

            self.features = nn.Sequential(*layers) # Features MobileNetv2

            # Última conv 1x1
            last_channel = _make_divisible(1280 * width_mult, 8) if width_mult > 1.0 else 1280

            self.conv_last = nn.Sequential(nn.Conv2d(in_c, last_channel, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(last_channel),
                nn.ReLU6(inplace=True))
            
            self.last_channel = last_channel

            self.pool = nn.AdaptiveAvgPool2d(1)
            self.classifier = nn.Linear(self.last_channel, num_classes)

        self.dropout = nn.Dropout(p=drop_out_rate)

    def forward(self, x):
        x = self.stem(x)
        x = self.features(x)

        if self.version == "v2":
            x = self.conv_last(x)

        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.classifier(x)
        return x
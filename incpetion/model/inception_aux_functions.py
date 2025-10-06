import torch
import torch.nn as nn

class ConvBNReLU(nn.Module):
    """
    Bloque: Conv2d (bias=False) -> BatchNorm2d -> ReLU.
    Útil para 1x1/3x3/5x5 con el padding correcto.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = 0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                      padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x: torch.Tensor):
        return self.net(x)

class InceptionV1Block(nn.Module):
    """
    Inception (GoogLeNet v1) canónico con 4 ramas en paralelo:
      - rama1: 1x1
      - rama2: 1x1 (reducción) -> 3x3
      - rama3: 1x1 (reducción) -> 5x5
      - rama4: 3x3 max-pool (stride=1, same) -> 1x1 (proyección)

    Parámetros (paper-like):
      in_channels:   canales de entrada al bloque
      out_1x1: #filtros de la rama 1 (1x1)
      red_3x3:  #filtros de reducción previa al 3x3
      out_3x3:  #filtros del 3x3
      red_5x5:   #filtros de reducción previa al 5x5
      out_5x5:  #filtros del 5x5
      pool_proj: #filtros de proyección 1x1 tras el max-pool

    Salida: concat canal (H y W se conservan).
    """
    def __init__(self,
        in_channels, out_1x1,
        red_3x3, out_3x3,
        red_5x5, out_5x5,
        pool_proj):
      
        super().__init__()

        # Rama 1: 1x1
        self.branch1 = ConvBNReLU(in_channels, out_1x1, kernel_size=1, padding=0)

        # Rama 2: 1x1 (reduce) -> 3x3 (same)
        self.branch2_reduce = ConvBNReLU(in_channels, red_3x3, kernel_size=1, padding=0)
        self.branch2_conv = ConvBNReLU(red_3x3, out_3x3, kernel_size=3, padding=1)

        # Rama 3: 1x1 (reduce) -> 5x5 (same)
        # En v1 el 5x5 es directamente 5x5 (padding=2). (En v2/v3 se reemplaza por 2x 3x3.)
        self.branch3_reduce = ConvBNReLU(in_channels, red_5x5, kernel_size=1, padding=0)
        self.branch3_conv = ConvBNReLU(red_5x5, out_5x5, kernel_size=5, padding=2)

        # Rama 4: 3x3 max-pool (stride=1, same) -> 1x1 (proj)
        self.branch4_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.branch4_proj = ConvBNReLU(in_channels, pool_proj, kernel_size=1, padding=0)

        #canales de salida totales
        self.out_channels = out_1x1 + out_3x3 + out_5x5 + pool_proj

    def forward(self, x: torch.Tensor):
        b1 = self.branch1(x)
        b2 = self.branch2_conv(self.branch2_reduce(x))
        b3 = self.branch3_conv(self.branch3_reduce(x))
        b4 = self.branch4_proj(self.branch4_pool(x))
        return torch.cat([b1, b2, b3, b4], dim=1)

class InceptionAuxHead(nn.Module):
    """
    Clasificador auxiliar (v1) — se engancha a las salidas de 4a y 4d.
    Estructura paper-like:
      AvgPool 5x5, stride=3  -> Conv 1x1 (128) -> ReLU
      -> Flatten -> FC (1024) -> ReLU -> Dropout(0.7) -> FC (num_classes)
    Nota: se asume que la entrada llega con tamaño espacial ~6x6 en STL original espera 14x14
    """
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))
        self.conv = nn.Conv2d(in_channels, 128, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

        # tras avgpool(5,3) sobre 14x14 -> 4x4 (aprox), 128*4*4=2048
        self.fc1 = nn.Linear(128 * 4 * 4, 1024)
        self.dropout = nn.Dropout(p=0.7)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x: torch.Tensor):
        x = self.avgpool(x)
        x = self.relu(self.conv(x))
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
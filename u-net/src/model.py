import torch.nn as nn
import torch

class ConvRelu(nn.Module):
    def __init__(self, in_c, out_c, k=3, s=1, p=1):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=k, stride=s, padding=p, bias=True),
            nn.ReLU(inplace=True))

    def forward(self, x):
        return self.net(x)
    

class MaxPool(nn.Module):
    def __init__(self, k=2, s=2, p=0):
        super().__init__()
        self.net = nn.MaxPool2d(kernel_size=k, stride=s, padding=p)

    def forward(self, x):
        return self.net(x)
    

class UnetEncoderLayer(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = ConvRelu(in_c, out_c)
        self.conv2 = ConvRelu(out_c, out_c)
        self.pool  = MaxPool()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        skip = x
        x = self.pool(x)
        return x, skip

  
class UpConv(nn.Module):
    def __init__(self, in_c, out_c, k=2, s=2, p=0):
        """
        in_c: canales de entrada (del nivel más profundo)
        out_c: canales de salida después de la up-conv
        k, s, p: kernel, stride, padding (por defecto 2x2 stride=2)
        """
        super().__init__()
        self.net = nn.ConvTranspose2d(in_c, out_c, kernel_size=k, stride=s, padding=p)

    def forward(self, x):
        return self.net(x)


class UnetDecoderLayer(nn.Module):
    """
    UpConv(duplica H,W) -> Concat con skip -> Conv3x3+ReLU -> Conv3x3+ReLU
    Args:
        in_c   : canales que entran desde el nivel más profundo del decoder
        skip_c : canales del skip correspondiente del encoder
        out_c  : canales deseados a la salida del bloque
    """
    def __init__(self, in_c, skip_c, out_c):
        super().__init__()
        self.up   = UpConv(in_c, out_c)
        self.conv1 = ConvRelu(out_c + skip_c, out_c)
        self.conv2 = ConvRelu(out_c, out_c)

    def forward(self, x, skip):
        x = self.up(x)

        if x.size(-2) != skip.size(-2) or x.size(-1) != skip.size(-1):
            dh = skip.size(-2) - x.size(-2)
            dw = skip.size(-1) - x.size(-1)
            skip = skip[..., dh//2 : skip.size(-2)-(dh-dh//2),
                        dw//2 : skip.size(-1)-(dw-dw//2)]

        x = torch.cat([skip, x], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x
    

class UNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=1, base=64):
        super().__init__()

        C = base
        self.enc = nn.ModuleList([
            UnetEncoderLayer(in_channels, C),   # 3 -> 64   (skip1)  H,W
            UnetEncoderLayer(C, 2*C),           # 64 -> 128 (skip2)  H/2
            UnetEncoderLayer(2*C, 4*C),         # 128->256  (skip3)  H/4
            UnetEncoderLayer(4*C, 8*C),         # 256->512  (skip4)  H/8
        ])

         # ---------- Bottleneck (doble conv) ----------
        self.bottleneck = nn.Sequential(
            ConvRelu(8*C, 16*C),                # 512 -> 1024
            ConvRelu(16*C, 16*C),               # 1024 -> 1024
        )

        # ---------- Decoder ----------
        self.dec = nn.ModuleList([
            UnetDecoderLayer(16*C, 8*C, 8*C),   # 1024 -> up -> concat skip4 (512) -> 512
            UnetDecoderLayer(8*C, 4*C, 4*C),    # 512  -> up -> concat skip3 (256) -> 256
            UnetDecoderLayer(4*C, 2*C, 2*C),    # 256  -> up -> concat skip2 (128) -> 128
            UnetDecoderLayer(2*C, C, C),        # 128  -> up -> concat skip1 (64)  -> 64
        ])

         # ---------- Salida ----------
        self.out_conv = nn.Conv2d(C, num_classes, kernel_size=1)

    def forward(self, x):
        skips = []

        # Encoder
        for layer in self.enc:
            x, skip = layer(x)
            skips.append(skip)      # [skip1, skip2, skip3, skip4]

        # Bottleneck
        x = self.bottleneck(x)
        # Decoder (usar skips en orden inverso)
        for layer, skip in zip(self.dec, reversed(skips)):
            x = layer(x, skip)

        logits = self.out_conv(x)
        return logits
    


import torch 
from model.MBConv import *
from model.compuder_scaler import *

class EfficientNet(nn.Module):
    """
    EfficientNet v1 paramétrico (B0..B7) con compound scaling.
    - Define B0 base mediante la tabla (t,k,c,n,s)
    - Aplica round_filters (ancho) y round_repeats (profundidad) con width/depth multipliers
    - Distribuye drop_connect linealmente a lo largo de todos los MBConv
    """

    # Tabla B0: (expand t, kernel k, out_channels c, repeats n, stride s)
    B0_CFG = [
        (1, 3,  16, 1, 1),
        (6, 3,  24, 2, 2),
        (6, 5,  40, 2, 2),
        (6, 3,  80, 3, 2),
        (6, 5, 112, 3, 1),
        (6, 5, 192, 4, 2),
        (6, 3, 320, 1, 1),]

    def __init__(self, num_classes: int = 101,
                 scaler: CompoundScaler = CompoundScaler(),
                 se_ratio: float = 0.25,
                 drop_connect_rate: float = 0.2,
                 dropout: float = 0.2,
                 stem_channels: int = 32,
                 head_channels: int = 1280,
                 channel_divisor: int = 8,
                 bn_eps: float = 1e-3,
                 bn_mom: float = 0.01): # IMPORTANTE: en PyTorch, 0.01 ≈ TF 0.99

        super().__init__()

        width_mult = scaler.width_mult()
        depth_mult = scaler.depth_mult()
        stem_c = round_filters(stem_channels, width_mult, channel_divisor)
        head_c = round_filters(head_channels, width_mult, channel_divisor)

        # Stem
        self.stem = ConvBNAct(3, stem_c, kernel_size=3, stride=2, groups=1,use_act=True, bn_eps=bn_eps, bn_mom=bn_mom)

        # MBConv stages
        total_blocks = 0
        repeats_per_stage = []
        for (t, k, c, n, s) in self.B0_CFG:
            n_round = round_repeats(n, depth_mult)
            repeats_per_stage.append(n_round)
            total_blocks += n_round

        # MBConv stages
        blocks = []
        block_idx = 0
        in_ch = stem_c

        for stage_idx, (t, k, c, n, s) in enumerate(self.B0_CFG):
            out_c = round_filters(c, width_mult, channel_divisor)
            repeats = repeats_per_stage[stage_idx]

            for i in range(repeats):
                stride_i = s if i == 0 else 1
                # escala lineal de drop_connect en profundidad
                denom = max(1, total_blocks - 1)
                drop_i = drop_connect_rate * (block_idx / denom)

                blocks.append(MBConv(
                    in_ch=in_ch,
                    out_ch=out_c,
                    kernel_size=k,
                    stride=stride_i,
                    expand_ratio=float(t),
                    se_ratio=se_ratio,
                    drop_connect=drop_i,
                    bn_eps=bn_eps,
                    bn_mom=bn_mom,
                    divisible_by=channel_divisor))

                in_ch = out_c
                block_idx += 1

        self.blocks = nn.Sequential(*blocks)

        # Head
        self.head = ConvBNAct(in_ch, head_c, kernel_size=1, stride=1, groups=1,
                              use_act=True, bn_eps=bn_eps, bn_mom=bn_mom)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(p=dropout, inplace=True) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(head_c, num_classes)

        # Pesos por defecto
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01); nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        x = self.pool(x)           # (N, C, 1, 1)
        x = torch.flatten(x, 1)    # (N, C)
        x = self.drop(x)
        x = self.fc(x)
        return x
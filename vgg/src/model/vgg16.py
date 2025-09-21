
import torch
import torch.nn as nn
from src.model.vgg_blocks import * 

class VGG16(nn.Module):
    def __init__(self, num_classes = 2, use_bn = False):
        super().__init__()
        self.features = nn.Sequential(
            VGGConvBlock(3,   64, num_convs=2, use_bn=use_bn, pool=True),   # 224 -> 112
            VGGConvBlock(64, 128, num_convs=2, use_bn=use_bn, pool=True),   # 112 -> 56
            VGGConvBlock(128,256, num_convs=3, use_bn=use_bn, pool=True),   # 56  -> 28
            VGGConvBlock(256,512, num_convs=3, use_bn=use_bn, pool=True),   # 28  -> 14
            VGGConvBlock(512,512, num_convs=3, use_bn=use_bn, pool=True),   # 14  -> 7
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            VGGDenseBlock(512 * 7 * 7, 4096, dropout=0.5),
            VGGDenseBlock(4096, 4096, dropout=0.5),
            nn.Linear(4096, num_classes))

        self.apply(init_vgg_weights)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
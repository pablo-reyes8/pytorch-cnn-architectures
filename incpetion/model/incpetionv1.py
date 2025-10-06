import torch.nn.functional as F
import torch
import torch.nn as nn
from model.inception_aux_functions import *


class GoogLeNetV1(nn.Module):
    """
    GoogLeNet / Inception v1 (ILSVRC 2014).
    - Stem inicial
    - 9 Inception blocks: (3a,3b) -> pool -> (4a..4e) -> pool -> (5a,5b)
    - Aux heads tras 4a y 4d (solo en entrenamiento)
    - Global average pool + Dropout(0.4) + FC final
    """
    def __init__(self, num_classes = 10, aux_logits = True):
        super().__init__()
        self.aux_logits = aux_logits

        # Stem 
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)   # STL10 96->48
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)                    # 48->24
        self.conv2 = nn.Conv2d(64, 64, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1, bias=False)  # mantiene HxW
        self.bn3 = nn.BatchNorm2d(192)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)   # 24->12 (si input=96)

        # Inception 3a, 3b 
        self.in3a = InceptionV1Block(192, 64, 96, 128, 16, 32, 32)   # out C = 256
        self.in3b = InceptionV1Block(256, 128, 128, 192, 32, 96, 64) # out C = 480
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # reduce HxW (6x6)

        # Inception 4a .. 4e 
        self.in4a = InceptionV1Block(480, 192, 96, 208, 16, 48, 64) # out C = 512
        self.in4b = InceptionV1Block(512, 160, 112, 224, 24, 64, 64) # out C = 512
        self.in4c= InceptionV1Block(512, 128, 128, 256, 24, 64, 64)  # out C = 512
        self.in4d = InceptionV1Block(512, 112, 144, 288, 32, 64, 64)  # out C = 528
        self.in4e = InceptionV1Block(528, 256, 160, 320, 32, 128, 128) # out C = 832
        self.pool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # reduce HxW (6x6)

        # Inception 5a, 5b 
        self.in5a = InceptionV1Block(832, 256, 160, 320, 32, 128, 128) # out C = 832
        self.in5b = InceptionV1Block(832, 384, 192, 384, 48, 128, 128) # out C = 1024

        # Aux Heads 
        if self.aux_logits:
            self.aux1 = InceptionAuxHead(in_channels=512, num_classes=num_classes)  # tras 4a
            self.aux2 = InceptionAuxHead(in_channels=528, num_classes=num_classes)  # tras 4d

        # Main Head 
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # global avg pool
        self.dropout = nn.Dropout(p=0.4)
        self.fc = nn.Linear(1024, num_classes)

        # Inicialización
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # Stem
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool2(x)

        # 3a, 3b
        x = self.in3a(x)
        x = self.in3b(x)
        x = self.pool3(x)

        # 4a
        x = self.in4a(x)
        aux1 = None

        if self.aux_logits and self.training:
            aux1 = self.aux1(x)  # después de 4a

        # 4b, 4c, 4d
        x = self.in4b(x)
        x = self.in4c(x)
        x = self.in4d(x)
        aux2 = None

        if self.aux_logits and self.training:
            aux2 = self.aux2(x)  # después de 4d

        # 4e
        x = self.in4e(x)
        x = self.pool4(x)

        # 5a, 5b
        x = self.in5a(x)
        x = self.in5b(x)

        # Head principal
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        logits = self.fc(x)

        if self.aux_logits and self.training:
            return logits, aux1, aux2

        return logits
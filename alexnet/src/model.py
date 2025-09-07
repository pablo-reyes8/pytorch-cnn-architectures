import torch.nn as nn
import torch


class ConvRelu(nn.Module):
    def __init__(self, in_c, out_c, k, s, p , use_lrn=False):
        super().__init__()

        layers = [
            nn.Conv2d(in_c, out_c, kernel_size=k, stride=s, padding=p, bias=True),
            nn.ReLU(inplace=True)]

        if use_lrn:
            # Hiperparámetros de AlexNet
            layers.append(nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2.0))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class MaxPool(nn.Module):
    def __init__(self, k=3, s=2, p=0):
        super().__init__()
        self.net = nn.MaxPool2d(kernel_size=k, stride=s, padding=p)

    def forward(self, x):
        return self.net(x)


class AlexNetClassifier(nn.Module):
    """
    Cabeza totalmente conectada de AlexNet.
    Por defecto espera features de tamaño 256×6×6 (flatten=9216) y produce logits.
    """
    def __init__(self, in_features=256*6*6, hidden=(4096, 4096), num_classes=10, p_drop=0.5):
        super().__init__()
        h1, h2 = hidden

        self.classifier = nn.Sequential(
            nn.Dropout(p_drop),
            nn.Linear(in_features, h1),
            nn.ReLU(inplace=True),

            nn.Dropout(p_drop),
            nn.Linear(h1, h2),
            nn.ReLU(inplace=True),

            nn.Linear(h2, num_classes))
        self._init_weights()

    def _init_weights(self):
        # AlexNet usaba ReLU; Kaiming normal conviene en FC también
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        x puede venir ya flatten (N, in_features) o como mapa (N, C, H, W).
        """
        if x.dim() == 4:                      # (N, C, H, W)
            x = torch.flatten(x, 1)           # (N, C*H*W)
        return self.classifier(x)
    

class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            ConvRelu(3,   96, 11, 4, 0, use_lrn=True),
            MaxPool(),
            ConvRelu(96, 256, 5, 1, 2, use_lrn=True),
            MaxPool(),
            ConvRelu(256, 384, 3, 1, 1),
            ConvRelu(384, 384, 3, 1, 1),
            ConvRelu(384, 256, 3, 1, 1),
            MaxPool())
        
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        self.classifier = AlexNetClassifier(
            in_features=256*6*6,
            hidden=(4096, 4096),
            num_classes=num_classes)

    def forward(self, x):             
        x = self.features(x)
        x = self.avgpool(x)         
        x = self.classifier(x)      
        return x
    

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
from tqdm.auto import tqdm


class ConvTanh(nn.Module):
    def __init__(self, in_c, out_c, k=5, s=1, p=0):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=k, stride=s, padding=p, bias=True),
            nn.Tanh())
        
    def forward(self, x):
        return self.net(x)


class SubsampleAvgPool(nn.Module):
    """Emula el 'subsampling' del paper con avg-pool + tanh."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Tanh())
        
    def forward(self, x):
        return self.net(x)
    

class LeNet5(nn.Module):
    """
    Entrada: (N, 1, 32, 32)
    Arquitectura: C1(6) -> S2 -> C3(16) -> S4 -> C5(120) -> F6(84) -> Out(10)
    Notas:
      - Usamos conexiones 'densas' en C3 (estándar moderno). 
      - Activación tanh para ser fiel al paper.
    """

    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            ConvTanh(1, 6, k=5),          # 32->28 , 1->6.  28×28×6
            SubsampleAvgPool(),           # 28->14.  14×14×6
            ConvTanh(6, 16, k=5),         # 14->10 , 6 -> 16.  10×10×16
            SubsampleAvgPool(),           # 10->5 5×5×16
            ConvTanh(16, 120, k=5)        # 5->1 , 16 -> 120.  1×1×120
        )
        self.classifier = nn.Sequential(
            nn.Linear(120, 84),  # 84 neuronas 
            nn.Tanh(), # Tanh Activation 
            nn.Linear(84, num_classes))  # 10 clases Softmax

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def init_tanh_xavier(module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('tanh'))
            if m.bias is not None:
                nn.init.zeros_(m.bias)



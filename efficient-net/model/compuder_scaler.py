import math
import torch.nn.functional as F
from dataclasses import dataclass
import torch

@dataclass
class CompoundScaler:
    """
    Escala compuesta de EfficientNet.
    Controla cómo crecen ancho, profundidad y resolución según φ.
    """
    def __init__(self, phi: int = 0, alpha: float = 1.2, beta: float = 1.1, gamma: float = 1.15):
        self.phi = phi
        self.alpha = alpha  # depth
        self.beta = beta    # width
        self.gamma = gamma  # resolution

    def width_mult(self):
        return self.beta ** self.phi

    def depth_mult(self):
        return self.alpha ** self.phi

    def res_mult(self):
        return self.gamma ** self.phi


class DynamicResize(torch.nn.Module):
    """
    Reescala un batch NCHW a la resolución target (H=W=target_size)
    mediante F.interpolate (mode='bilinear' por defecto).
    """
    def __init__(self, mode: str = "bilinear", align_corners: bool = False):
        super().__init__()
        self.mode = mode
        self.align_corners = align_corners

    @torch.no_grad()
    def forward(self, x: torch.Tensor, target_size: int) -> torch.Tensor:
        # x: [N, C, H, W]
        if x.shape[-1] == target_size and x.shape[-2] == target_size:
            return x
        return F.interpolate(x, size=(target_size, target_size),mode=self.mode, align_corners=self.align_corners)
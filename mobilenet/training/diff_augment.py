import torch
import torch.nn as nn

class DiffAugment(nn.Module):
    """
    Pequeñas augmentations diferenciables:
    - Flip horizontal
    - Jitter de brillo y contraste
    - Cutout (un cuadrado negro por imagen con cierta probabilidad)

    Se aplica sobre tensores [B, C, H, W].
    """
    def __init__(
        self,
        p_flip: float = 0.5,
        p_brightness: float = 0.5,
        max_brightness_delta: float = 0.2,
        p_contrast: float = 0.5,
        max_contrast_scale: float = 0.2,
        p_cutout: float = 0.5,
        cutout_frac: float = 0.25,  # tamaño del cuadrado como fracción de H y W
    ):

        super().__init__()
        self.p_flip = p_flip
        self.p_brightness = p_brightness
        self.max_brightness_delta = max_brightness_delta
        self.p_contrast = p_contrast
        self.max_contrast_scale = max_contrast_scale
        self.p_cutout = p_cutout
        self.cutout_frac = cutout_frac

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, H, W]
        """
        if not self.training:
            return x

        B, C, H, W = x.shape
        device = x.device

        # Flip horizontal
        if self.p_flip > 0.0:
            flip_mask = (torch.rand(B, 1, 1, 1, device=device) < self.p_flip)
            x_flipped = torch.flip(x, dims=[3])
            x = torch.where(flip_mask, x_flipped, x)

        # Brillo: x + delta
        if self.p_brightness > 0.0 and self.max_brightness_delta > 0.0:
            mask_b = (torch.rand(B, 1, 1, 1, device=device) < self.p_brightness)
            delta = (torch.rand(B, 1, 1, 1, device=device) * 2 - 1) * self.max_brightness_delta
            x = x + mask_b * delta

        # Contraste: x * scale
        if self.p_contrast > 0.0 and self.max_contrast_scale > 0.0:
            mask_c = (torch.rand(B, 1, 1, 1, device=device) < self.p_contrast)
            scale = 1.0 + (torch.rand(B, 1, 1, 1, device=device) * 2 - 1) * self.max_contrast_scale
            x = x * (1.0 + mask_c * (scale - 1.0))

        # Cutout
        if self.p_cutout > 0.0 and self.cutout_frac > 0.0:
            mask_size_h = int(self.cutout_frac * H)
            mask_size_w = int(self.cutout_frac * W)
            mask_size_h = max(1, mask_size_h)
            mask_size_w = max(1, mask_size_w)

            for i in range(B):
                if torch.rand(1, device=device) < self.p_cutout:
                    cy = torch.randint(0, H, (1,), device=device).item()
                    cx = torch.randint(0, W, (1,), device=device).item()

                    y1 = max(0, cy - mask_size_h // 2)
                    y2 = min(H, cy + mask_size_h // 2)
                    x1 = max(0, cx - mask_size_w // 2)
                    x2 = min(W, cx + mask_size_w // 2)

                    x[i, :, y1:y2, x1:x2] = 0.0  

        return x
import torch

def denormalize(img_tensor, mean=[0.48293063044548035, 0.44492557644844055, 0.3957090973854065], 
                std=[0.2592383325099945, 0.25327032804489136, 0.2598187029361725]):
    """
    Invierte Normalize: x * std + mean
    Soporta [C,H,W] ó [N,C,H,W].
    """
    if img_tensor.dim() == 4:
        C = img_tensor.size(1)
    else:
        C = img_tensor.size(0)
    mean_t = torch.tensor(mean[:C], device=img_tensor.device).view(1 if img_tensor.dim()==4 else C, 1, 1)
    std_t  = torch.tensor(std[:C],  device=img_tensor.device).view(1 if img_tensor.dim()==4 else C, 1, 1)
    return img_tensor * std_t + mean_t

def _to_numpy_img(img_chw):
    """
    Convierte un tensor [C,H,W] en imagen numpy HxWxC (RGB o 1 canal).
    Clampa a [0,1] para mostrar.
    """
    img = img_chw.detach().cpu().clamp(0, 1)
    if img.size(0) == 1:
        return img.squeeze(0).numpy()
    else:
        return img.permute(1, 2, 0).numpy()
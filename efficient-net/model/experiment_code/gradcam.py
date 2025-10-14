
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

def cleanup_all_hooks(model: torch.nn.Module):
    # remueve forward, forward_pre, backward y full_backward hooks residuales
    for m in model.modules():
        for attr in (
            "_forward_hooks", "_forward_pre_hooks",
            "_backward_hooks", "_backward_pre_hooks",
            "_full_backward_hooks"):
            d = getattr(m, attr, None)

            if isinstance(d, dict) and len(d):
                for h in list(d.values()):
                    try: h.remove()
                    except: pass
                d.clear()
        

def get_module_by_name(root: torch.nn.Module, dotted: str) -> torch.nn.Module:
    """
    Resuelve rutas tipo 'blocks.15.depthwise.conv' dentro del modelo.
    Ej: get_module_by_name(model_b0, "head.conv")
    """
    m = root
    for p in dotted.split("."):
        if p.isdigit():
            m = m[int(p)]
        else:
            m = getattr(m, p)
    return m

def _denorm_img(x: torch.Tensor,
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)) -> torch.Tensor:
    """
    x: (C,H,W) en [-? ?] normalizado IMAGENET → des-normaliza a [0,1] aprox.
    """
    mean = torch.tensor(mean, device=x.device)[:, None, None]
    std  = torch.tensor(std,  device=x.device)[:, None, None]
    y = x * std + mean
    return y.clamp(0, 1)


def colorize_cam(cam_01):  
    """Devuelve (H,W,3) con un colormap sencillo (jet)."""
    heat = cam_01[0,0].cpu().numpy()
    cmap = plt.cm.jet(heat)[:,:,:3]  
    return cmap

def show_cam_overlay(x_chnw, cam_01, alpha=0.35):
    """x_chnw: (C,H,W) normalizado IMAGENET; cam_01: (1,1,H,W) [0,1]."""
    base = _denorm_img(x_chnw).cpu().numpy()    
    base = np.transpose(base, (1,2,0))            
    cmap = colorize_cam(cam_01)                   
    out  = (1-alpha)*base + alpha*cmap
    out  = np.clip(out, 0, 1)
    plt.figure(figsize=(5,5))
    plt.imshow(out)
    plt.axis("off")
    plt.tight_layout()
    plt.show()

class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module | str):
        self.model = model.eval()
        if isinstance(target_layer, str):
            self.target_layer = get_module_by_name(model, target_layer)
        else:
            self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        def fwd_hook(module, inp, out):
            self.activations = out
            if isinstance(self.activations, torch.Tensor):
                self.activations.retain_grad()  

        def bwd_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        self.hook_fwd = self.target_layer.register_forward_hook(fwd_hook)
        self.hook_bwd = self.target_layer.register_full_backward_hook(bwd_hook)

    @torch.inference_mode(False)
    def __call__(self, x: torch.Tensor, class_idx: int | None = None,use_relu: bool = True, normalize: bool = True):
        
        assert x.ndim == 4 and x.size(0) == 1, "Pasa un batch de tamaño 1."
        self.model.zero_grad(set_to_none=True)

        logits = self.model(x) 
        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())
        score = logits[0, class_idx]

        # backward
        score.backward(retain_graph=True)

        A = self.activations        
        dY = self.gradients          

        assert A is not None and dY is not None, "Activaciones/gradientes no capturados; cambia de capa objetivo."

        weights = dY.mean(dim=(2,3), keepdim=True)  
        cam = (weights * A).sum(dim=1, keepdim=True) 

        if use_relu:
            cam = torch.relu(cam)

        if normalize:
            eps = 1e-8
            cam_min = cam.amin(dim=(2,3), keepdim=True)
            cam_max = cam.amax(dim=(2,3), keepdim=True)
            cam = (cam - cam_min) / (cam_max - cam_min + eps)

        H, W = x.shape[-2], x.shape[-1]
        cam_up = torch.nn.functional.interpolate(cam, size=(H,W),
                                                 mode="bilinear", align_corners=False)
        return cam_up.detach(), class_idx

    def remove_hooks(self):
        for h in (self.hook_fwd, self.hook_bwd):
            try:
                h.remove()
            except Exception:
                pass

def overlay_cam_on_image(
    x: torch.Tensor,      
    cam: torch.Tensor,       
    alpha: float = 0.35) -> torch.Tensor:
    """
    Devuelve overlay como (C,H,W) en [0,1].
    No fuerza colormap; usa cam como máscara (gris) y mezcla alpha simple.
    """
    assert x.ndim == 3 and cam.ndim == 4
    base = _denorm_img(x)                  
    heat = cam[0, 0]                        
    heat3 = heat.expand_as(base)             
    out = (1 - alpha) * base + alpha * heat3
    return out.clamp(0, 1)


def show_cam_colorized(x_chw, cam, alpha=0.45, cmap_name="jet"):
    """
    x_chw: (C,H,W) normalizado IMAGENET
    cam  : (1,1,H,W) [0,1]
    """
    base = _denorm_img(x_chw).cpu().numpy()      
    base = np.transpose(base, (1,2,0))           
    heat = cam[0,0].cpu().numpy()
    cmap = plt.get_cmap(cmap_name)
    heat_color = cmap(heat)[:,:,:3]               
    out = (1 - alpha) * base + alpha * heat_color
    out = np.clip(out, 0, 1)
    return out


# Use case: 

# toma un batch y usa hasta 10 imágenes
# xb, yb = next(iter(val_loader))
# n_show = min(10, xb.size(0))
# xb = xb[:n_show].to(device)

# 5 capas a diferentes profundidades (ajústalas si quieres otras)
# targets = [
    "blocks.3.depthwise.conv",
    "blocks.7.depthwise.conv",
    "blocks.11.project.conv",
    "blocks.15.depthwise.conv",
    "head.conv"]


# for i in range(n_show):
    x1 = xb[i:i+1]
    x1_chw = xb[i].detach().cpu()

    fig, axes = plt.subplots(1, len(targets), figsize=(len(targets)*3.2, 3.2))
    for j, tgt in enumerate(targets):
        gc = GradCAM(model_b0, tgt)
        cam, cls = gc(x1, class_idx=None)
        vis = show_cam_colorized(x1_chw, cam.cpu(), alpha=0.45, cmap_name="jet")
        axes[j].imshow(vis)
        axes[j].set_title(tgt, fontsize=9)
        axes[j].axis("off")
        gc.remove_hooks()
    plt.tight_layout()
    plt.show()
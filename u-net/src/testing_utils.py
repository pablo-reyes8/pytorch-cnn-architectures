import numpy as np, torch, matplotlib.pyplot as plt
from scipy import ndimage as ndi
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_curve, auc
from sklearn.manifold import TSNE
try:
    import umap
    HAVE_UMAP = True
except:
    HAVE_UMAP = False
from scipy.spatial.distance import directed_hausdorff
import torch.nn as nn
import torch.nn.functional as F


#################### Segmentation Performance Evaluation ######################### 

def unnormalize_img_1(x_chw, mean, std):
    """
    x_chw: tensor [C,H,W] normalizado
    mean,std: tu normalización por canal (tuplas/listas de len C)
    return: np.float32 en [0,1]
    """
    x = x_chw.detach().cpu().float().clone()
    for c in range(x.size(0)):
        x[c] = x[c] * std[c] + mean[c]
    x = x.clamp(0, 1)
    return x.permute(1,2,0).numpy()

def get_logits(out):
    """Convierte la salida del modelo a un Tensor de logits."""
    if isinstance(out, (tuple, list)):
        return out[0]
    if isinstance(out, dict):
        return out.get('out', next(iter(out.values())))
    return out

def clear_all_hooks(module):
    # borra hooks de este módulo
    if hasattr(module, "_forward_hooks"):       module._forward_hooks.clear()
    if hasattr(module, "_forward_pre_hooks"):   module._forward_pre_hooks.clear()
    if hasattr(module, "_backward_hooks"):      module._backward_hooks.clear()
    # recursivo
    for m in module.children():
        clear_all_hooks(m)

def viz_overlay_errors(xb, out_or_logits, yb, thr=0.5, mean=None, std=None, titles=True):
    """
    Visualiza imagen, mapa de probabilidad y mapa de errores (R=FP, G=TP, B=FN).

    xb:  [1,C,H,W]
    out_or_logits: Tensor logits [1,1,H,W] o salida cruda del modelo (tuple/dict/Tensor)
    yb:  [1,1,H,W] o [1,H,W] con {0,1}
    thr: umbral de probabilidad para binarizar (binario)
    mean/std: tu normalización (p.ej. ImageNet) para desnormalizar la imagen
    """
    assert xb.size(0) == 1, "Pasa batch de tamaño 1 para visualizar"
    x = xb[0]

    # Imagen a [H,W,3] en [0,1]
    if mean is not None and std is not None:
        img = unnormalize_img_1(x, mean, std)
    else:
        img = x.detach().cpu().permute(1,2,0).float().numpy()
        # si C=1, replicamos canales para mostrar en RGB
        if img.shape[2] == 1:
            img = np.repeat(img, 3, axis=2)
        vmin, vmax = img.min(), img.max()
        img = (img - vmin) / (vmax - vmin + 1e-8)
    img = np.clip(img, 0, 1)

    # Logits -> probas
    logits = get_logits(out_or_logits)
    prob = torch.sigmoid(logits)[0,0].detach().cpu().numpy()

    # Ground truth a [H,W] uint8
    y = yb[0]
    if y.dim() == 3:   # [1,H,W] -> [H,W]
        y = y[0]
    gt = y.detach().cpu().numpy().astype(np.uint8)

    # Predicción binaria
    pred = (prob > thr).astype(np.uint8)

    # Contornos (opcional)
    edge_pred = np.logical_xor(pred, ndi.binary_erosion(pred))
    edge_gt   = np.logical_xor(gt,   ndi.binary_erosion(gt))

    # Mapa de errores RGB
    err = np.zeros((*pred.shape, 3), dtype=float)
    err[...,0] = (pred == 1) & (gt == 0)   # FP → rojo
    err[...,1] = (pred == 1) & (gt == 1)   # TP → verde
    err[...,2] = (pred == 0) & (gt == 1)   # FN → azul

    # Plots
    plt.figure(figsize=(14,4))
    ax = plt.subplot(1,3,1); ax.imshow(img); ax.axis('off')
    if titles: ax.set_title("Imagen")

    ax = plt.subplot(1,3,2); ax.imshow(img); ax.imshow(prob, alpha=0.5); ax.axis('off')
    if titles: ax.set_title("Probabilidad (sigmoid)")

    ax = plt.subplot(1,3,3); ax.imshow(img*0.6); ax.imshow(err, alpha=0.6); ax.axis('off')
    if titles: ax.set_title("Errores  (R=FP, G=TP, B=FN)")
    plt.tight_layout(); plt.show()



#################### Precision–Recall and ROC Curve Analysis ######################### 


def plot_pr_roc_from_logits(logits, yb, mask_valid=None):
    """
    logits: [B,1,H,W], yb: [B,1,H,W] o [B,H,W] {0,1}
    mask_valid: opcional [B,H,W] o [B,1,H,W] (True donde evaluar)
    """
    probs = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
    y = yb.detach().cpu()
    if y.dim()==3: y = y.unsqueeze(1)
    y = y.numpy().reshape(-1)

    if mask_valid is not None:
        m = mask_valid
        if isinstance(m, torch.Tensor): m = m.detach().cpu().numpy()
        m = m.reshape(-1).astype(bool)
        probs, y = probs[m], y[m]

    p, r, _ = precision_recall_curve(y, probs)
    ap = average_precision_score(y, probs)
    fpr, tpr, _ = roc_curve(y, probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1); plt.plot(r, p); plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(f"PR (AP={ap:.3f})"); plt.grid(True, alpha=.3)
    plt.subplot(1,2,2); plt.plot(fpr, tpr); plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"ROC (AUC={roc_auc:.3f})"); plt.grid(True, alpha=.3)
    plt.tight_layout(); plt.show()



####################   Image-level Overlap Metrics (Dice & IoU)  ####################


def dice_per_image_from_logits(logits, yb, thr=0.5, eps=1e-7):
    """
    retorna lista de Dice por imagen
    """
    probs = torch.sigmoid(logits)
    if yb.dim()==3: yb = yb.unsqueeze(1)
    y = yb.float()
    pred = (probs > thr).float()

    inter = (pred*y).sum(dim=(1,2,3))
    sums = pred.sum(dim=(1,2,3)) + y.sum(dim=(1,2,3))
    dice = (2*inter + eps)/(sums + eps)                 # [B]
    return dice.detach().cpu().numpy()

def iou_per_image_from_logits(logits, yb, thr=0.5, eps=1e-7):
    probs = torch.sigmoid(logits)
    if yb.dim()==3: yb = yb.unsqueeze(1)
    y = yb.float()
    pred = (probs > thr).float()
    inter = (pred*y).sum(dim=(1,2,3))
    union = pred.sum(dim=(1,2,3)) + y.sum(dim=(1,2,3)) - inter
    iou = (inter + eps)/(union + eps)
    return iou.detach().cpu().numpy()

def plot_hist_metrics(dices, ious):
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1); plt.hist(dices, bins=20); plt.title(f"Dice (mean={np.mean(dices):.3f})")
    plt.subplot(1,2,2); plt.hist(ious,  bins=20); plt.title(f"IoU  (mean={np.mean(ious):.3f})")
    plt.tight_layout(); plt.show()



####################   Pixel-level Calibration Analysis  ####################


def calibration_curve_pixels(logits, yb, n_bins=10):
    probs = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
    y = yb.detach().cpu()
    if y.dim()==3: y = y.unsqueeze(1)
    y = y.numpy().reshape(-1)

    bins = np.linspace(0,1,n_bins+1)
    binids = np.digitize(probs, bins)-1
    conf, acc = [], []
    for b in range(n_bins):
        m = (binids==b)
        if m.sum()==0:
            conf.append(np.nan); acc.append(np.nan)
        else:
            conf.append(probs[m].mean())
            acc.append(y[m].mean())
    conf, acc = np.array(conf), np.array(acc)
    plt.figure(figsize=(4.5,4.5))
    plt.plot([0,1],[0,1],'--',alpha=.5)
    plt.plot(conf, acc, marker='o')
    plt.xlabel("Confianza promedio (p)"); plt.ylabel("Frecuencia empírica")
    plt.title("Calibración (pixel-level)"); plt.grid(True, alpha=.3); plt.show()


####################   Convolutional Feature Map Visualization #################### 


def _first_tensor(o):
    if isinstance(o, torch.Tensor):
        return o
    if isinstance(o, (list, tuple)):
        for v in o:
            t = _first_tensor(v)
            if t is not None:
                return t
    if isinstance(o, dict):
        for v in o.values():
            t = _first_tensor(v)
            if t is not None:
                return t
    return None

def visualize_feature_maps(model, layer, xb, num_maps=16):
    activ = {}

    def hook_fn(m, i, o):
        # toma el primer Tensor que encuentre en o (soporta tuple/list/dict)
        import torch
        def first_tensor(x):
            if isinstance(x, torch.Tensor): return x
            if isinstance(x, (list, tuple)):
                for v in x:
                    t = first_tensor(v)
                    if t is not None: return t
            if isinstance(x, dict):
                for v in x.values():
                    t = first_tensor(v)
                    if t is not None: return t
            return None

        t = first_tensor(o)
        if t is None:
            raise TypeError(f"No pude extraer un Tensor de la salida de {m.__class__.__name__}. Tipo={type(o)}")
        activ['maps'] = t.detach().cpu()

    handle = layer.register_forward_hook(hook_fn)

    was_training = model.training
    try:
        model.eval()
        xb = xb.to(next(model.parameters()).device, non_blocking=True)
        with torch.no_grad():
            _ = model(xb)
    finally:
        handle.remove()
        if was_training: model.train()

    maps = activ['maps']
    if maps.ndim == 4: maps = maps[0]      # [C,H,W]
    elif maps.ndim == 3: pass
    elif maps.ndim == 2: maps = maps.unsqueeze(0)
    else: raise ValueError(f"Dim no soportada: {maps.shape}")

    import matplotlib.pyplot as plt
    n = min(num_maps, maps.size(0))
    plt.figure(figsize=(n, 1.6))
    for i in range(n):
        ax = plt.subplot(1, n, i+1)
        fmap = maps[i]; mn, mx = fmap.min(), fmap.max()
        fmap = (fmap - mn) / (mx - mn + 1e-6)
        ax.imshow(fmap, cmap='gray'); ax.axis('off')
    plt.suptitle(f"{n} feature maps de {layer.__class__.__name__}", y=1.02, fontsize=10)
    plt.tight_layout(); plt.show()

def pick_layer(block, kind= nn.Conv2d, idx=0):
    """
    Retorna la subcapa número idx de tipo `kind` dentro de `block`.
    También imprime los nombres encontrados para que puedas elegir.
    """
    found = [(n, m) for n, m in block.named_modules() if isinstance(m, kind)]
    if not found:
        raise ValueError(f"No se encontró ninguna subcapa de tipo {kind} en {block.__class__.__name__}")
    print("Encontré estas capas:", [n for n, _ in found])
    return found[idx][1]



####################  Bottleneck Representation Analysis (UMAP/t-SNE) #################### 



def collect_bottleneck(model, loader, device, hook_layer):
    """
    hook_layer: módulo en el bottleneck
    retorna: X [N, D] flattened, y opcional si loader lo provee
    """
    feats = []
    ys = []
    def hook_fn(m,i,o):
        feats.append(o.detach().cpu())
    h = hook_layer.register_forward_hook(hook_fn)

    model.eval()
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            _ = model(xb)
            if isinstance(yb, torch.Tensor):
                ys.append(yb.clone())
    h.remove()
    X = torch.cat([f.flatten(1) for f in feats], dim=0).numpy()
    Y = torch.cat(ys, dim=0).numpy() if ys else None
    return X, Y

def plot_embedding_2d(X, Y=None, method="umap", n=5000, title="Bottleneck embedding"):
    Xs = X[:n]
    if method=="umap" and HAVE_UMAP:
        reducer = umap.UMAP(n_components=2)
        Z = reducer.fit_transform(Xs)
    else:
        Z = TSNE(n_components=2, init='random', learning_rate='auto').fit_transform(Xs)
    plt.figure(figsize=(5,5))
    if Y is None:
        plt.scatter(Z[:,0], Z[:,1], s=2)
    else:
        y = Y[:n].reshape(-1)
        sc = plt.scatter(Z[:,0], Z[:,1], c=y, s=2, cmap='tab10')
        plt.colorbar(sc, fraction=0.046)
    plt.title(title); plt.axis('off'); plt.show()




####################  Boundary Quality Metrics (Boundary-F1 and Hausdorff Distance) #################### 


def boundary_map(mask):
    mask = mask.astype(bool)
    return np.logical_xor(mask, ndi.binary_erosion(mask))

def boundary_f1(pred, gt, tol=2):
    """
    F1 sobre bordes con tolerancia en pixeles (afín a BFScore).
    pred, gt: binarios [H,W]
    """
    bp = boundary_map(pred); bg = boundary_map(gt)
    # dilatación para tolerancia
    se = ndi.generate_binary_structure(2,1)
    dp = ndi.binary_dilation(bp, structure=se, iterations=tol)
    dg = ndi.binary_dilation(bg, structure=se, iterations=tol)

    tp_p = (bp & dg).sum();  tp_g = (bg & dp).sum()
    p = bp.sum();            r = bg.sum()
    prec = tp_p / (p + 1e-7); rec = tp_g / (r + 1e-7)
    f1 = 2*prec*rec/(prec+rec+1e-7)
    return f1, prec, rec

def hausdorff_distance(pred, gt):
    """ Hausdorff simétrico aproximado (usa puntos de borde). """
    bp = np.argwhere(boundary_map(pred))
    bg = np.argwhere(boundary_map(gt))
    if len(bp)==0 or len(bg)==0:
        return np.nan
    d1 = directed_hausdorff(bp, bg)[0]
    d2 = directed_hausdorff(bg, bp)[0]
    return max(d1, d2)


####################  Occlusion Sensitivity Analysis (ΔDice) #################### 


def unnormalize_img(x, mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)):
    """
    x: Tensor [3,H,W] normalizado -> retorna np.ndarray [H,W,3] en [0,1]
    Ajusta mean/std a lo que usaste en tus transforms.
    """
    x = x.detach().cpu().clone()
    for c in range(min(3, x.shape[0])):
        x[c] = x[c]*std[c] + mean[c]
    x = x.clamp(0,1).permute(1,2,0).numpy()
    return x

@torch.no_grad()
def occlusion_sensitivity(
    model, xb, yb, patch=24, stride=12, thr=0.5, device='cuda',
    mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225), overlay=True):

    """
    Devuelve heatmap (H,W) con la caída de Dice al ocluir parches.
    Superpone el heatmap sobre la imagen des-normalizada para visualizar.
    """

    assert xb.size(0) == 1
    model.eval()
    xb = xb.to(device)
    yb = yb.to(device).float()
    if yb.dim() == 3: yb = yb.unsqueeze(1)  # [1,1,H,W]
    H, W = xb.shape[-2:]

    # --- Predicción base y Dice base ---
    base_logits = model(xb)
    base_prob   = torch.sigmoid(base_logits)[0,0]            # [H,W]
    base_pred   = (base_prob > thr).float()
    inter       = (base_pred * yb[0,0]).sum()
    denom       = base_pred.sum() + yb[0,0].sum()
    eps = 1e-7
    base_dice   = (2*inter + eps) / (denom + eps)

    rows = list(range(0, H - patch + 1, stride))
    cols = list(range(0, W - patch + 1, stride))
    heat_coarse = torch.zeros(1, 1, len(rows), len(cols), device=device)

    # Relleno por canal: media de la propia imagen (mejor que poner 0)
    fill = xb.mean(dim=(-1,-2), keepdim=True)   # [1,C,1,1]

    for ri, i in enumerate(rows):
        for cj, j in enumerate(cols):
            xb_occ = xb.clone()
            xb_occ[..., i:i+patch, j:j+patch] = fill  # oclusión "natural"
            p = torch.sigmoid(model(xb_occ))[0,0]
            pred = (p > thr).float()
            inter = (pred * yb[0,0]).sum()
            denom = pred.sum() + yb[0,0].sum()
            dice  = (2*inter + eps) / (denom + eps)
            drop  = torch.clamp(base_dice - dice, min=0.0)
            heat_coarse[0,0,ri,cj] = drop

    # --- Interpolar a resolución completa para visualización suave ---
    heat = F.interpolate(
        heat_coarse, size=(H, W), mode='bilinear', align_corners=False)[0,0].detach().cpu().numpy()

    # Normalizar heat para colorbar agradable
    if heat.max() > 0:
        heat_vis = heat / (heat.max() + 1e-8)
    else:
        heat_vis = heat

    # --- Visualización ---
    plt.figure(figsize=(12,4))

    plt.subplot(1,3,1); plt.title("Imagen")
    img = unnormalize_img(xb[0], mean=mean, std=std)
    plt.imshow(img); plt.axis('off')

    plt.subplot(1,3,2); plt.title("Pred prob")
    plt.imshow(base_prob.detach().cpu(), vmin=0, vmax=1)
    plt.axis('off'); plt.colorbar(fraction=0.046)

    # Heatmap puro
    plt.subplot(1,3,3); plt.title("Occlusion ΔDice")
    plt.imshow(heat_vis)
    plt.axis('off'); plt.colorbar(fraction=0.046)

    plt.tight_layout(); plt.show()

    # Overlay opcional (útil para reportes)
    if overlay:
        plt.figure(figsize=(5,5))
        plt.title("Overlay ΔDice")
        plt.imshow(img)
        plt.imshow(heat_vis, alpha=0.5, cmap='magma')
        plt.axis('off'); plt.colorbar(fraction=0.046)
        plt.tight_layout(); plt.show()

    return heat





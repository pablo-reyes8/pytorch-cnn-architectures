import time
import math
from collections import Counter
from typing import Tuple, Sequence, Optional

import torch
from torch.utils.data import DataLoader
import numpy as np

def _batch_stats(x) :
    """
    Devuelve (mean, std) por canal para un batch [N, C, H, W], en el mismo device.
    """
    assert x.dim() == 4, f"Esperaba 4D, obtuve {x.shape}"
    N, C, H, W = x.shape
    # promedio sobre N,H,W
    mean = x.float().mean(dim=(0, 2, 3))
    std  = x.float().std(dim=(0, 2, 3), unbiased=False)
    return mean, std

@torch.no_grad()
def check_loaders_basic(train_loader,val_loader,
    class_names,
    num_classes,*,
    img_size = 224,
    mode = "multiclass",      # 'binary' | 'multiclass'
    sample_batches = 3,
    tol_shape = 0,
    verbose = True):

    """
    Chequeos básicos sobre DataLoaders:
      - Formas del batch (N,C,H,W), H=W=img_size
      - Rango/consistencia de labels y num_classes
      - Coherencia class_names
      - Stats post-normalización (aprox mean~0, std~1 si usas z-score)
    """

    assert mode in {"binary", "multiclass"}
    assert num_classes == len(class_names), "num_classes != len(class_names)"

    def _peek(dl, name):
        it = iter(dl)
        xb, yb = next(it)
        if verbose:
            print(f"[{name}] batch0 shape: {tuple(xb.shape)}, labels shape: {tuple(yb.shape)}")
        N, C, H, W = xb.shape
        assert C == 3, f"Se esperaban 3 canales, got {C}"
        if tol_shape == 0:
            assert H == img_size and W == img_size, f"H,W != {img_size}"
        else:
            assert abs(H - img_size) <= tol_shape and abs(W - img_size) <= tol_shape, \
                f"H,W no cerca de {img_size}±{tol_shape}"


        y_min, y_max = int(yb.min().item()), int(yb.max().item())
        if verbose:
            print(f"[{name}] label range in first batch: [{y_min}, {y_max}]")

        if mode == "binary":
            assert num_classes == 2, "Binary mode requiere num_classes=2"
            assert {0, 1}.issuperset(set(yb.unique().tolist())), "Labels no son {0,1}"
        else:
            assert num_classes >= 2, "Multiclass requiere >=2 clases"
            assert (0 <= y_min) and (y_max < num_classes), "Labels fuera de rango [0, num_classes-1]"


        m, s = _batch_stats(xb)
        if verbose:
            print(f"[{name}] post-norm mean (C): {m.detach().cpu().numpy().round(3)}")
            print(f"[{name}] post-norm  std  (C): {s.detach().cpu().numpy().round(3)}")


        if (s < 1e-6).any():
            raise ValueError(f"[{name}] std ~ 0 detectada en algún canal (posible normalización errónea).")

    _peek(train_loader, "train")
    _peek(val_loader,   "val")

    def _approx_class_hist(dl, name):
        counts = Counter()
        n_seen = 0
        for i, (_, yb) in enumerate(dl):
            counts.update(yb.tolist())
            n_seen += yb.numel()
            if i + 1 >= sample_batches:
                break
        if verbose:
            top = sorted(counts.items())
            print(f"[{name}] approx class hist (primeros {n_seen} labels): {top}")

    _approx_class_hist(train_loader, "train")
    _approx_class_hist(val_loader,   "val")

    if verbose:
        print("[OK] check_loaders_basic finalizado.")

@torch.no_grad()
def check_shuffling_and_speed(train_loader: DataLoader,*,
    probe_batches = 5, verbose= True):

    """
    Comprueba si hay indicios de shuffling (comparando primeras posiciones de dos iteraciones)
    y mide velocidad promedio de iteración.
    """

    it1 = iter(train_loader)
    xb1, yb1 = next(it1)
    it2 = iter(train_loader)
    xb2, yb2 = next(it2)

    same_first_labels = torch.equal(yb1[:10], yb2[:10])
    if verbose:
        print(f"[shuffle] primeras 10 labels iguales en 2 corridas? {same_first_labels} (False ~ buen indicio)")

    t0 = time.time()
    n = 0
    for i, (xb, yb) in enumerate(train_loader):
        _ = xb.shape, yb.shape
        n += 1
        if i + 1 >= probe_batches:
            break
    dt = time.time() - t0
    if verbose:
        print(f"[speed] {probe_batches} batches en {dt:.3f}s → {probe_batches/dt:.2f} it/s aprox.")

    if verbose:
        print("[OK] check_shuffling_and_speed finalizado.")



@torch.no_grad()
def check_no_nan_inf_in_sample(loader,*,sample_batches = 3,verbose= True):
    """
    Verifica que no haya NaN/Inf en algunos batches (features y labels).
    """

    c_feat = c_lab = 0
    for i, (xb, yb) in enumerate(loader):
        if torch.isnan(xb).any() or torch.isinf(xb).any():
            raise ValueError("Se detectaron NaN/Inf en features.")
        if torch.isnan(yb).any() or torch.isinf(yb).any():
            raise ValueError("Se detectaron NaN/Inf en labels.")
        c_feat += xb.numel(); c_lab += yb.numel()
        if i + 1 >= sample_batches:
            break
    if verbose:
        print(f"[OK] check_no_nan_inf_in_sample: {sample_batches} batches limpios "
              f"({c_feat} feats, {c_lab} labels revisados).")
        

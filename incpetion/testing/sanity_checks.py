
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
import random
import os

def set_seed(seed=1337):
    random.seed(seed); np.random.seed(seed)
    os.environ["PYTHONHASHSEED"]=str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False

@torch.no_grad()
def sanity_batch(loader, n=1):
    X,y = next(iter(loader))
    print(f"X: {tuple(X.shape)}, dtype={X.dtype}, range=({X.min():.3f},{X.max():.3f})")
    print(f"y: {tuple(y.shape)}, dtype={y.dtype}, classes=[{int(y.min())}..{int(y.max())}]")
    if n>1:
        for _ in range(n-1): next(iter(loader))

@torch.no_grad()
def sanity_forward(model, loader, device="cuda"):
    model.eval().to(device)
    X,y = next(iter(loader))
    X = X.to(device)
    out = model(X)
    logits = out[0] if isinstance(out, tuple) else out
    ok = torch.isfinite(logits).all().item()
    print("Forward OK. Logits finitos:", ok, "| shape:", tuple(logits.shape))


def sanity_backward_step(model, loader, device="cuda"):
    model.train().to(device)
    opt = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    X,y = next(iter(loader)); X,y = X.to(device), y.to(device)
    opt.zero_grad()
    out = model(X); logits = out[0] if isinstance(out, tuple) else out
    loss = torch.nn.functional.cross_entropy(logits, y)
    loss.backward()
    nonzero = sum((p.grad is not None and torch.any(p.grad!=0).item()) for p in model.parameters())
    opt.step()
    print(f"Loss: {loss.item():.4f} | Parámetros con grad≠0: {nonzero}")


def sanity_nans(model):
    import math
    n_w = sum(torch.isnan(p).any().item() or torch.isinf(p).any().item() for p in model.parameters())
    n_g = sum((p.grad is not None) and (torch.isnan(p.grad).any().item() or torch.isinf(p.grad).any().item())
              for p in model.parameters())
    print(f"NaN/Inf -> pesos:{n_w} | grads:{n_g}")


def sanity_overfit_one_batch(model, loader, steps=200, lr=1e-2, device="cuda"):
    model.train().to(device)
    X,y = next(iter(loader)); X,y = X.to(device), y.to(device)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    losses=[]
    for t in range(steps):
        opt.zero_grad()
        out = model(X); logits = out[0] if isinstance(out, tuple) else out
        loss = torch.nn.functional.cross_entropy(logits, y)
        loss.backward(); opt.step()
        losses.append(loss.item())
        if (t+1)%50==0: print(f"step {t+1}/{steps} loss {losses[-1]:.4f}")
    print("↓ Loss inicial:", losses[0], "→ final:", losses[-1])
    return losses


@torch.no_grad()
def sanity_modes(model, loader, device="cuda"):
    X,_ = next(iter(loader)); X = X.to(device)
    model.train().to(device)
    out_train = model(X)
    model.eval()
    out_eval  = model(X)
    print("train() devuelve tuple?:", isinstance(out_train, tuple))
    print("eval()  devuelve tuple?:", isinstance(out_eval, tuple))

from collections import Counter
def sanity_class_balance(loader, num_classes=10):
    cnt = Counter()
    for _,y in loader:
        cnt.update(y.tolist())
        break  
    total = sum(cnt.values())
    print("Distribución (primer batch):")
    for c in range(num_classes):
        p = 100.0*cnt[c]/total if total>0 else 0
        print(f"  clase {c}: {cnt[c]} ({p:.1f}%)")


def sanity_grad_magnitudes(model):
    mags = []
    for n,p in model.named_parameters():
        if p.grad is not None:
            g = p.grad.detach().abs().mean().item()
            mags.append((n, g))
    mags.sort(key=lambda x: x[1])
    print("Gradientes más chicos:")
    for n,g in mags[:5]: print(f"{n:50s} {g:.3e}")
    print("Gradientes más grandes:")
    for n,g in mags[-5:]: print(f"{n:50s} {g:.3e}")


def sanity_lr(optimizer):
    lrs = [pg["lr"] for pg in optimizer.param_groups]
    print("LRs:", lrs)


@torch.no_grad()
def sanity_inference_speed(model, loader, device="cuda", warmup=5, iters=20):
    import time
    model.eval().to(device)
    X,_ = next(iter(loader)); X = X.to(device)
    for _ in range(warmup): _ = model(X)
    if device.startswith("cuda"): torch.cuda.synchronize()
    t0=time.perf_counter()
    for _ in range(iters):
        _ = model(X)
        if device.startswith("cuda"): torch.cuda.synchronize()
    dt=time.perf_counter()-t0
    print(f"{(dt/iters)*1000:.2f} ms/batch | {((dt/iters)/X.size(0))*1000:.2f} ms/img")


@torch.no_grad()
def sanity_determinism(model_fn, loader, seed=123, device="cuda"):
    set_seed(seed)
    m1 = model_fn().to(device).eval()
    X,_ = next(iter(loader)); X = X.to(device)
    p1 = m1(X).argmax(1).cpu().numpy()
    set_seed(seed)
    m2 = model_fn().to(device).eval()
    p2 = m2(X).argmax(1).cpu().numpy()
    print("Predicciones idénticas con misma semilla?:", np.array_equal(p1, p2))


def sanity_aux_losses(model, loader, device="cuda"):
    model.train().to(device)
    X,y = next(iter(loader)); X,y = X.to(device), y.to(device)
    out = model(X)
    if isinstance(out, tuple):
        logits, a1, a2 = out
        l0 = torch.nn.functional.cross_entropy(logits, y).item()
        l1 = torch.nn.functional.cross_entropy(a1, y).item()
        l2 = torch.nn.functional.cross_entropy(a2, y).item()
        print(f"Main:{l0:.3f} Aux1:{l1:.3f} Aux2:{l2:.3f}  |  Total ~ {l0 + 0.3*(l1+l2):.3f}")
    else:
        print("El modelo no devuelve auxiliares (eval() o aux_logits=False).")

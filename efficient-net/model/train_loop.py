from torch.nn.functional import cross_entropy
from sklearn.metrics import f1_score
import torch
from model.cnn_utils import * 
import torch.nn.functional as F
from model.compuder_scaler import *

def make_grad_scaler(device_type: str, enabled: bool = True):
    """
    Crea un GradScaler sin disparar deprecations.
    - PyTorch nuevo: torch.amp.GradScaler(device_type='cuda'|'cpu', enabled=...)
    - PyTorch intermedio: torch.amp.GradScaler('cuda'|'cpu', enabled=...)  # firma antigua
    """
    # API moderna/unificada
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            # Firma nueva
            return torch.amp.GradScaler(device_type=device_type, enabled=enabled)
        except TypeError:
            # Firma vieja
            return torch.amp.GradScaler(device_type, enabled=enabled)

    # Ruta 2: muy antiguo (sin torch.amp)
    if device_type == "cuda":
        from torch.cuda.amp import GradScaler as CudaGradScaler
        return CudaGradScaler(enabled=enabled)


def train_efficientnet_multiclass(
    model,
    train_loader,
    optimizer,
    scheduler=None,
    num_epochs = 10,
    device= "cuda",
    label_smoothing: float = 0.0, # Para mejorar Cross Entropy
    grad_clip: float | None = 5.0,
    use_amp = True,
    # parámetros para compound scaling de resolución
    base_input_size: int = 200,     # 200x200 food 101
    scaler=None,                    # instancia CompoundScaler(phi=0..7)
    channel_divisor: int = 8,       # Controla el redondeo de canales en make_divisible / round_filters
    progressive: bool = False,      # si True: incrementa resolución progresivamente por época
):
    """
    Entrena EfficientNet (o cualquier CNN sin logits auxiliares), aplicando compound scaling de
    RESOLUCIÓN en el batch con DynamicResize. Mantiene AMP, label smoothing y grad clipping.

    - base_input_size: tamaño con el que llegan tus batches (p.ej., 200).
    - scaler: CompoundScaler(alpha,beta,gamma,phi). Usamos gamma**phi para la resolución target.
    - progressive: si True, incrementa suavemente la resolución desde base_input_size a target
      a lo largo de las épocas (útil para acelerar warmup).
    """
    device = torch.device(device)
    torch.set_float32_matmul_precision("high")
    model.to(device)
    model.to(memory_format=torch.channels_last)
    model.train()

    # AMP scaler
    scaler_amp = make_grad_scaler(device.type, enabled=use_amp)


    resizer = DynamicResize()
    # Calcula resolución objetivo a partir de gamma^phi (si hay scaler)
    if scaler is not None:
        res_mult = (scaler.gamma ** scaler.phi)
    else:
        res_mult = 1.0

    target_size_full = round_resolution(base_input_size, res_mult, divisor=channel_divisor)

    history = {"train_loss": [], "train_acc": [], "train_f1_macro": []}

    autocast_dtype = torch.float16 if device.type == "cuda" else torch.bfloat16
    for epoch in range(num_epochs):
        epoch_loss, correct, total = 0.0, 0, 0
        y_true_train, y_pred_train = [], []

        # Si entrenamiento progresivo: interpola linealmente el tamaño por época
        if progressive:
            t = (epoch + 1) / num_epochs
            t_smooth = 0.5 * (1 - torch.cos(torch.tensor(t * 3.14159265))).item()
            curr_size = int(round(base_input_size + t_smooth * (target_size_full - base_input_size)))
            curr_size = max(channel_divisor, int(round(curr_size / channel_divisor) * channel_divisor))
        else:
            curr_size = target_size_full

        for X, y in train_loader:
            X = X.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            y = y.to(device, non_blocking=True)

            # compound scaling de resolución en el batch
            if (X.shape[-2] != curr_size) or (X.shape[-1] != curr_size):
                X = resizer(X, target_size=curr_size) # escalamos las imaganes segun el factor

            optimizer.zero_grad(set_to_none=True)


            use_autocast = use_amp and device.type in {"cuda", "cpu"}
            autocast_kwargs = {"device_type": device.type,"dtype": torch.float16 if device.type == "cuda" else torch.bfloat16,
                              "enabled": use_autocast}
            if device.type == "cpu":
                autocast_kwargs["dtype"] = torch.bfloat16


            with torch.autocast(**autocast_kwargs):
                logits = model(X)
                loss = F.cross_entropy(logits, y, label_smoothing=label_smoothing) # Definimos directamente Corssentropy aca
                preds = logits.argmax(dim=1)

            # Backprop
            if scaler_amp and scaler_amp.is_enabled():
                scaler_amp.scale(loss).backward()
                if grad_clip is not None:
                    scaler_amp.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                scaler_amp.step(optimizer)
                scaler_amp.update()
            else:
                loss.backward()
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                optimizer.step()

            # métricas
            bs = y.size(0)
            epoch_loss += loss.item() * bs
            correct += (preds == y).sum().item()
            total += bs
            y_true_train.extend(y.detach().cpu().tolist())
            y_pred_train.extend(preds.detach().cpu().tolist())

        if scheduler is not None:
            scheduler.step()

        avg_train_loss = epoch_loss / total
        train_acc = correct / total
        train_f1 = f1_score(y_true_train, y_pred_train, average="macro")

        history["train_loss"].append(avg_train_loss)
        history["train_acc"].append(train_acc)
        history["train_f1_macro"].append(train_f1)

        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"size:{curr_size} | Loss:{avg_train_loss:.4f} | Acc:{train_acc:.3f} | F1:{train_f1:.3f}")

    return history



##################### TRAIN LOOP ADVANCE FOR DEPURATION #####################

from torch.nn.utils import clip_grad_norm_
import math

@torch.no_grad()
def _tensor_stats(x, name="tensor"):
    finite = torch.isfinite(x).all().item()
    _min = x.min().item() if finite else float('nan')
    _max = x.max().item() if finite else float('nan')
    _mean = x.mean().item() if finite else float('nan')
    _std  = x.std().item()  if finite else float('nan')
    print(f"[stats] {name}: finite={finite} shape={tuple(x.shape)} "
          f"min={_min:.4f} max={_max:.4f} mean={_mean:.4f} std={_std:.4f}")

def _current_lr(optimizer):
    lrs = [pg['lr'] for pg in optimizer.param_groups if 'lr' in pg]
    return lrs[0] if lrs else None

def _safe_label_smoothing(ls: float) -> float:
    return float(max(0.0, min(0.999, ls)))

def train_efficientnet_multiclass_advanced(
    model,
    train_loader,
    optimizer,
    scheduler=None,
    num_epochs=10,
    device="cuda",
    label_smoothing: float = 0.0,
    grad_clip: float | None = 5.0,
    use_amp=True,
    # compound scaling 
    base_input_size: int = 200,
    scaler=None,                   
    channel_divisor: int = 8,
    progressive: bool = False,
    # depuración
    detect_anomaly: bool = False,   #  traza exacta del op que genera NaN (lento)
    print_every: int = 50,          # cada cuántos pasos imprimir stats
):
    """
    Loop robusto con diagnósticos de NaN/Inf.
    - AMP y channels_last solo en CUDA.
    - Chequeos y early-exit con reportes detallados si aparece NaN/Inf.
    """
    device = torch.device(device)
    model.to(device).float()
    if device.type == "cuda":
        model.to(memory_format=torch.channels_last)

    model.train()
    torch.set_float32_matmul_precision("high")
    if detect_anomaly:
        torch.autograd.set_detect_anomaly(True)

    # AMP solo en CUDA
    use_autocast = (device.type == "cuda") and bool(use_amp)
    scaler_amp = make_grad_scaler(device.type, enabled=use_autocast) 

    # Resolución objetivo  (CompoundScaler)
    res_mult = (scaler.gamma ** scaler.phi) if (scaler is not None) else 1.0
    target_size_full = round_resolution(base_input_size, res_mult, divisor=channel_divisor)
    resizer = DynamicResize()

    ls = _safe_label_smoothing(label_smoothing)

    history = {"train_loss": [], "train_acc": [], "train_f1_macro": []}

    for epoch in range(num_epochs):
        epoch_loss, correct, total = 0.0, 0, 0
        y_true_train, y_pred_train = [], []

        # tamaño progresivo (cosine) si se pide
        if progressive:
            t = (epoch + 1) / num_epochs
            t_smooth = 0.5 * (1 - math.cos(math.pi * t))
            curr_size = int(round(base_input_size + t_smooth * (target_size_full - base_input_size)))
            curr_size = max(channel_divisor, int(round(curr_size / channel_divisor) * channel_divisor))
        else:
            curr_size = target_size_full

        for step, (X, y) in enumerate(train_loader):
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            if device.type == "cuda":
                X = X.to(memory_format=torch.channels_last)

            if (X.shape[-2] != curr_size) or (X.shape[-1] != curr_size):
                X = resizer(X, target_size=curr_size)

            # Chequeo rápido de entrada
            if not torch.isfinite(X).all():
                print(f"[ERROR] Entrada con NaN/Inf en epoch={epoch} step={step}")
                _tensor_stats(X, "X")
                return history

            optimizer.zero_grad(set_to_none=True)

            try:
                if use_autocast:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        logits = model(X)
                        loss = F.cross_entropy(logits, y, label_smoothing=ls)
                else:
                    logits = model(X)     
                    loss= F.cross_entropy(logits, y, label_smoothing=ls)

                # Diagnóstico de finitud
                if not torch.isfinite(loss):
                    print(f"[NaN/Inf LOSS] epoch={epoch} step={step} size={curr_size} lr={_current_lr(optimizer)}")
                    _tensor_stats(X, "X (post-resize)")
                    _tensor_stats(logits, "logits")
                    # distribuciones por clase del batch
                    with torch.no_grad():
                        unique, counts = y.unique(return_counts=True)
                        print("[batch] etiquetas únicas:", dict(zip(unique.tolist(), counts.tolist())))
                    return history

                preds = logits.argmax(dim=1)

                # Backward + step
                if scaler_amp.is_enabled():
                    scaler_amp.scale(loss).backward()
                    if grad_clip is not None:
                        scaler_amp.unscale_(optimizer)
                        clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                    scaler_amp.step(optimizer)
                    scaler_amp.update()
                else:
                    loss.backward()
                    if grad_clip is not None:
                        clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                    optimizer.step()

                # métrica
                bs = y.size(0)
                epoch_loss += loss.item() * bs
                correct += (preds == y).sum().item()
                total += bs
                y_true_train.extend(y.detach().cpu().tolist())
                y_pred_train.extend(preds.detach().cpu().tolist())

                # logging ocasional
                if (step % max(1, print_every) == 0):
                    print(f"[e{epoch+1} s{step}] size={curr_size} "
                          f"lr={_current_lr(optimizer)} loss={loss.item():.4f} "
                          f"acc_batch={(preds==y).float().mean().item():.3f}")

            except RuntimeError as e:
                print(f"[RuntimeError] epoch={epoch} step={step}: {e}")
                print("Sugerencias: use_amp=False, baja LR, revisa DynamicResize y normalización.")
                return history

        if scheduler is not None:
            scheduler.step()

        avg_train_loss = epoch_loss / max(1, total)
        train_acc = correct / max(1, total)
        train_f1 = f1_score(y_true_train, y_pred_train, average="macro")

        history["train_loss"].append(avg_train_loss)
        history["train_acc"].append(train_acc)
        history["train_f1_macro"].append(train_f1)

        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"size:{curr_size} | lr={_current_lr(optimizer)} | "
              f"Loss:{avg_train_loss:.4f} | Acc:{train_acc:.3f} | F1:{train_f1:.3f}")

    return history
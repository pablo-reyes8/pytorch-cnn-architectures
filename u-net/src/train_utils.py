from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
import torch


def _dice_coeff(pred, target, eps=1e-7):
    # pred y target binarios [B,1,H,W] en {0,1}
    inter = (pred * target).sum(dim=(1,2,3))
    union = pred.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3))
    dice = (2*inter + eps) / (union + eps)
    return dice.mean().item()


def _mean_iou_mc(pred, target, num_classes, eps=1e-7):
    # multiclase: pred [B,H,W] en {0..C-1}, target idem
    ious = []
    for c in range(num_classes):
        pred_c = (pred == c)
        targ_c = (target == c)
        inter = (pred_c & targ_c).sum().float()
        union = (pred_c | targ_c).sum().float()
        if union > 0:
            ious.append(((inter + eps) / (union + eps)).item())
    return sum(ious)/len(ious) if ious else 0.0


def train_epoch_seg(dataloader, model, optimizer, criterion,num_classes=1, device=None, amp=False, desc="Train"):
    """
    Entrena 1 época un modelo de segmentación (U-Net u otro).

    Args
    ----
    dataloader: DataLoader que retorna (xb, yb)
        - xb: [B, C, H, W]
        - yb:
            * binario: [B, 1, H, W] o [B, H, W] con {0,1}
            * multiclase: [B, H, W] con enteros en [0..num_classes-1]
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    criterion:
        * binario: nn.BCEWithLogitsLoss
        * multiclase: nn.CrossEntropyLoss
    num_classes: int
        - 1 para binario; >=2 para multiclase
    device: torch.device o str
    amp: bool -> usar mixed precision si hay CUDA disponible
    desc: str -> texto para la barra de progreso

    Returns
    -------
    dict con loss promedio, pixel accuracy y métrica (Dice o mIoU).
    """

    # --- device ---
    if device is None:
        device = next(model.parameters()).device
    elif isinstance(device, str): # Si es string ('cuda'/'cpu'), conviértelo a torch.device
        device = torch.device(device)

    use_amp = bool(amp and torch.cuda.is_available() and device.type == "cuda") # Activa AMP solo si: el usuario lo pidió, hay CUDA y el device es 'cuda'
    scaler = GradScaler(enabled=use_amp) # Crea el GradScaler (activo solo si use_amp=True)

    # --- helpers ---
    def _ensure_binary_target(y):
        # Asegura forma [B,1,H,W] y tipo float en {0,1}
        if y.dim() == 3:  # [B,H,W]
            y = y.unsqueeze(1)
        return y.float()

    def _dice_batch(pred, target, eps=1e-7):
        # pred/target: [B,1,H,W] binarios {0,1}
        inter = (pred * target).sum(dim=(1,2,3))
        sums  = pred.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3))
        dice = (2 * inter + eps) / (sums + eps)  # Fórmula de Dice
        return dice.mean()  # Promedio sobre el batch

    # --- estado ---
    model.train()
    running_loss = 0.0
    running_metric = 0.0
    correct_pix = 0
    n_pix = 0
    n_samples = 0

    pbar = tqdm(dataloader, total=len(dataloader), leave=False, desc=desc)

    for xb, yb in pbar:
        xb = xb.to(device, non_blocking=True) # Sube el batch a device (GPU/CPU)
        yb = yb.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True) # Limpia gradientes (más eficiente con set_to_none=True)

        with autocast(enabled=use_amp): # Forward en precisión mixta si aplica
            logits = model(xb)   # Predicciones [B,1,H,W] (bin) o [B,C,H,W] (mc)

            if num_classes == 1:
                yb_bin = _ensure_binary_target(yb)
                loss = criterion(logits, yb_bin)   # BCEWithLogitsLoss espera float en {0,1}
            else:
                loss = criterion(logits, yb.long())    # CrossEntropyLoss: target entero [B,H,W]

        if use_amp:   # Backward + step con AMP
            scaler.scale(loss).backward()  # Escala loss para evitar underflow en gradientes FP16
            scaler.step(optimizer)
            scaler.update()  # Actualiza dinámicamente el factor de escala
        else:
            loss.backward()
            optimizer.step()  # Backprop normal en FP32

        # --- acumuladores de loss ---
        bs = xb.size(0)
        running_loss += float(loss.detach().cpu()) * bs
        n_samples += bs

        # --- métricas rápidas ---
        with torch.no_grad():    # No necesitamos gradientes para métricas
            if num_classes == 1:
                probs = torch.sigmoid(logits)      # Convierte logits a probabilidades
                pred = (probs > 0.5).float()
                yb_bin = _ensure_binary_target(yb)

                correct_pix += (pred == yb_bin).sum().item() # Píxeles correctos en el batch
                n_pix += yb_bin.numel() # Píxeles totales en el batch

                dice = _dice_batch(pred, yb_bin)        # Dice medio del batch
                running_metric += float(dice.cpu()) * bs
                metric_name = "Dice"
            else:
                pred = logits.argmax(dim=1)             # Clase más probable por píxel [B,H,W]
                correct_pix += (pred == yb).sum().item()
                n_pix       += yb.numel()

                miou = _mean_iou_mc(pred, yb, num_classes)  # debes tenerla implementada
                running_metric += float(miou) * bs # Píxeles correctos
                metric_name = "mIoU"

        # --- progreso ---
        pix_acc = 100.0 * correct_pix / max(1, n_pix) # Exactitud por píxel acumulada (%)
        pbar.set_postfix(loss=f"{running_loss/max(1,n_samples):.4f}",pix_acc=f"{pix_acc:.2f}%",**{metric_name: f"{running_metric/max(1,n_samples):.3f}"})

    # --- resumen de época ---
    epoch_loss = running_loss / max(1, n_samples) # Loss final por muestra en la época
    pix_acc = 100.0 * correct_pix / max(1, n_pix) # Pixel accuracy final
    final_metric = running_metric / max(1, n_samples) # Métrica final (Dice o mIoU)

    print(f"Train - loss: {epoch_loss:.4f} | pix_acc: {pix_acc:.2f}% | {metric_name}: {final_metric:.3f}")
    return {'loss': epoch_loss, 'pix_acc': pix_acc, metric_name: final_metric}


def evaluate_seg(dataloader, model, criterion,
                 num_classes=1, device=None, amp=True, desc="Val"):
    """
    Evalúa 1 época (sin gradientes). Reporta loss promedio por muestra,
    pixel accuracy y Dice (binario) o mIoU (multiclase).
    """
    # --- device ---
    if device is None:
        device = next(model.parameters()).device
    elif isinstance(device, str):
        device = torch.device(device)

    use_amp = bool(amp and torch.cuda.is_available() and device.type == "cuda")

    def _ensure_binary_target(y):
        if y.dim() == 3:  # [B,H,W] -> [B,1,H,W]
            y = y.unsqueeze(1)
        return y.float()

    def _dice_batch(pred, target, eps=1e-7):
        inter = (pred * target).sum(dim=(1,2,3))
        sums  = pred.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3))
        return ((2*inter + eps) / (sums + eps)).mean()

    model.eval()
    running_loss = 0.0
    running_metric = 0.0
    correct_pix = 0
    n_pix = 0
    n_samples = 0

    pbar = tqdm(dataloader, total=len(dataloader), leave=False, desc=desc)

    with torch.no_grad():
        for xb, yb in pbar:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            with autocast(enabled=use_amp):
                logits = model(xb)
                if num_classes == 1:
                    yb_bin = _ensure_binary_target(yb)
                    loss = criterion(logits, yb_bin)
                else:
                    loss = criterion(logits, yb.long())

            bs = xb.size(0)
            running_loss += float(loss.detach().cpu()) * bs
            n_samples += bs

            if num_classes == 1:
                probs = torch.sigmoid(logits)            # [B,1,H,W]
                pred  = (probs > 0.5).float()
                yb_bin = _ensure_binary_target(yb)

                correct_pix += (pred == yb_bin).sum().item()
                n_pix       += yb_bin.numel()

                dice = _dice_batch(pred, yb_bin)
                running_metric += float(dice.cpu()) * bs
                metric_name = "Dice"
            else:
                pred = logits.argmax(dim=1)              # [B,H,W]
                correct_pix += (pred == yb).sum().item()
                n_pix       += yb.numel()

                miou = _mean_iou_mc(pred, yb, num_classes)
                running_metric += float(miou) * bs
                metric_name = "mIoU"

            # Postfix estable
            avg_loss = running_loss / max(1, n_samples)
            pix_acc  = 100.0 * correct_pix / max(1, n_pix)
            avg_met  = running_metric / max(1, n_samples)
            pbar.set_postfix(loss=f"{avg_loss:.4f}",pix_acc=f"{pix_acc:.2f}%",**{metric_name: f"{avg_met:.3f}"})

    epoch_loss = running_loss / max(1, n_samples)
    pix_acc    = 100.0 * correct_pix / max(1, n_pix)
    final_metric = running_metric / max(1, n_samples)
    print(f"Val   - loss: {epoch_loss:.4f} | pix_acc: {pix_acc:.2f}% | {metric_name}: {final_metric:.3f}")
    return {'loss': epoch_loss, 'pix_acc': pix_acc, metric_name: final_metric}



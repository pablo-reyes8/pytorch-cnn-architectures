from tqdm.auto import tqdm
import torch
import torch.nn.functional as F
from model import *


def train_epoch_classification(dataloader, model, optimizer, criterion, device=None, amp=False, desc="Train"):
    """
    Entrena una época (multiclase) con barra de progreso por batch.
    Muestra % completado y métricas en vivo (loss, acc@1, acc@3).
    """
    if device is None:
        device = next(model.parameters()).device
    elif isinstance(device, str):
        device = torch.device(device)
    device_type = device.type  # 'cuda' | 'cpu' | 'mps'


    model.train()
    running_loss = 0.0
    n_samples = 0
    correct1 = 0.0
    correct3 = 0.0

    use_amp = (amp and device_type == 'cuda')
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    pbar = tqdm(dataloader, total=len(dataloader), leave=False, desc=desc)

    for xb, yb in pbar:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.amp.autocast('cuda', enabled=True):
                logits = model(xb)           
                loss = criterion(logits, yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        bs = xb.size(0)
        running_loss += loss.item() * bs
        n_samples += bs

        # métricas por batch (top-1 y top-3)
        with torch.no_grad():
            maxk = 3
            _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)
            correct = pred.eq(yb.view(-1, 1).expand_as(pred))  # [N,3]
            correct1 += correct[:, :1].reshape(-1).float().sum().item()
            correct3 += correct[:, :3].reshape(-1).float().sum().item()

        epoch_loss = running_loss / n_samples
        acc1 = 100.0 * (correct1 / n_samples)
        acc3 = 100.0 * (correct3 / n_samples)

        pbar.set_postfix(loss=f"{epoch_loss:.4f}", acc1=f"{acc1:.2f}%", acc3=f"{acc3:.2f}%")

    print(f"Train - loss: {epoch_loss:.4f} | acc@1: {acc1:.2f}% | acc@3: {acc3:.2f}%")
    return {'loss': epoch_loss, 'acc1': acc1, 'acc3': acc3}


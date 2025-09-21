import torch

from src.training.train_utils import *

def train_epoch_classification(dataloader,
    model,optimizer,
    criterion,*,device=None,
    amp=False,desc="Train",
    scheduler=None, pos_label: int = 1 ):

    """
    Entrena UNA época para clasificación (2 clases).
    - Sin barra de progreso.
    - Devuelve e imprime: loss, accuracy (%) y F1 (binario).
    - Si `scheduler` no es None, hace `scheduler.step()` al final de la época.
    """

    if device is None:
        device = next(model.parameters()).device
    elif isinstance(device, str):
        device = torch.device(device)
    device_type = device.type

    model.train()
    running_loss = 0.0
    n_samples = 0

    correct = 0
    tp = fp = fn = tn = 0

    use_amp = (amp and device_type == 'cuda')
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    for xb, yb in dataloader:
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

        with torch.no_grad():
            preds = logits.argmax(dim=1)

            correct += (preds == yb).sum().item()

            if pos_label in (0, 1):
                if pos_label == 0:
                    preds_bin = (preds == 0).long()
                    yb_bin    = (yb == 0).long()
                else:
                    preds_bin = (preds == 1).long()
                    yb_bin    = (yb == 1).long()

                _tp, _fp, _fn, _tn = _update_binary_counts(preds_bin, yb_bin)
                tp += _tp; fp += _fp; fn += _fn; tn += _tn
            else:
                _tp, _fp, _fn, _tn = _update_binary_counts(preds, yb)
                tp += _tp; fp += _fp; fn += _fn; tn += _tn

    epoch_loss = running_loss / max(1, n_samples)
    acc = 100.0 * correct / max(1, n_samples)
    precision, recall, f1 = _safe_prec_recall_f1(tp, fp, fn)

    if scheduler is not None:
        if hasattr(scheduler, 'step') and scheduler.__class__.__name__ == 'ReduceLROnPlateau':
            scheduler.step(epoch_loss)
        else:
            scheduler.step()

    print(f"{desc} - loss: {epoch_loss:.4f} | acc: {acc:.2f}% | F1: {f1*100:.2f}% "
          f"(P: {precision*100:.2f}%, R: {recall*100:.2f}%)")

    return {
        'loss': epoch_loss,
        'acc': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn}
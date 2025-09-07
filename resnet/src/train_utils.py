import torch
import torch.nn.functional as F
from tqdm.auto import tqdm


@torch.no_grad()
def _topk_accuracies(logits, targets, ks=(1,)):
    """Devuelve un dict con top-k accuracies en %."""
    maxk = max(ks)
    batch_size = targets.size(0)

    _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)  # [N, maxk]
    pred = pred.t()                                                # [maxk, N]
    correct = pred.eq(targets.view(1, -1).expand_as(pred))         # [maxk, N]

    out = {}
    for k in ks:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        out[f"top{k}"] = (correct_k.mul_(100.0 / batch_size)).item()
    return out


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


def evaluate_classification(dataloader, model, criterion, device=None, return_preds=False, compute_top3=True):
    """
    Evalúa clasificación multiclase. Métricas: loss, acc@1 (y acc@3 opcional).

    Args:
        dataloader: iterable de (inputs, targets)
        model: nn.Module -> logits [N, C]
        criterion: nn.CrossEntropyLoss()
        device: opcional
        return_preds: si True, devuelve (pred_labels, true_labels, probs)
        compute_top3: si True, calcula acc@3 además de acc@1

    Returns:
        Si return_preds=False:
            dict {'loss': float, 'acc1': float, 'acc3': float (si aplica)}
        Si return_preds=True:
            (metrics_dict, pred_labels (list[int]), true_labels (list[int]), probs (list[list[float]]))
    """
    if device is None:
        device = next(model.parameters()).device


    model.eval()
    running_loss = 0.0
    n_samples = 0

    total_correct1 = 0.0
    total_correct3 = 0.0

    preds_all = []
    labels_all = []
    probs_all = []

    with torch.no_grad():
        for xb, yb in dataloader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            logits = model(xb)                      # [N, C]
            loss = criterion(logits, yb)
            running_loss += loss.item() * xb.size(0)
            n_samples += xb.size(0)

            # Métricas
            topks = (1, 3) if compute_top3 else (1,)
            topk = _topk_accuracies(logits, yb, ks=topks)
            total_correct1 += topk['top1'] * xb.size(0) / 100.0
            if compute_top3:
                total_correct3 += topk['top3'] * xb.size(0) / 100.0

            if return_preds:
                probs = F.softmax(logits, dim=1)            # [N, C]
                _, pred = torch.max(logits, dim=1)          # [N]
                preds_all.extend(pred.cpu().tolist())
                labels_all.extend(yb.cpu().tolist())
                probs_all.extend(probs.cpu().tolist())

    metrics = {
        'loss': running_loss / n_samples,
        'acc1': 100.0 * (total_correct1 / n_samples)}
    
    if compute_top3:
        metrics['acc3'] = 100.0 * (total_correct3 / n_samples)

    if not return_preds:
        print_str = f"Val  - loss: {metrics['loss']:.4f} | acc@1: {metrics['acc1']:.2f}%"
        if compute_top3:
            print_str += f" | acc@3: {metrics['acc3']:.2f}%"
        print(print_str)
        return metrics
    else:
        return metrics, preds_all, labels_all, probs_all
    


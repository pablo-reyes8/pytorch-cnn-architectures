import torch
import torch.nn as nn



@torch.no_grad()
def accuracy_topk(output: torch.Tensor, target: torch.Tensor,topk=(1, 3, 5)):
    """
    Calcula top-k accuracy para cada k en topk.
    output: logits [B, num_classes]
    target: labels [B]
    """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, dim=1, largest=True, sorted=True) 
    pred = pred.t()  

    # comparar con labels
    correct = pred.eq(target.view(1, -1).expand_as(pred)) 

    res = {}
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res[f"top{k}"] = (correct_k / batch_size).item()
    return res

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_one_epoch(model: nn.Module, dataloader,
                    optimizer: torch.optim.Optimizer, device: torch.device,
                    criterion: nn.Module, diff_augment: nn.Module = None):
    """
    Entrena una época y devuelve métricas agregadas.

    Retorna un dict con:
      - loss
      - top1
      - top3
      - top5
    """

    model.train()
    if diff_augment is not None:
        diff_augment.train()

    running_loss = 0.0
    running_top1 = 0.0
    running_top3 = 0.0
    running_top5 = 0.0
    n_samples = 0

    for imgs, labels in dataloader:
        imgs = imgs.to(device, non_blocking=True)

        labels = labels.to(device, non_blocking=True)
        batch_size = imgs.size(0)

        if diff_augment is not None:
            imgs = diff_augment(imgs)

        # Forward
        outputs = model(imgs)
        loss = criterion(outputs, labels)

        # Backward
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # Métricas
        with torch.no_grad():
            accs = accuracy_topk(outputs, labels, topk=(1, 3, 5))
            running_loss += loss.item() * batch_size
            running_top1 += accs["top1"] * batch_size
            running_top3 += accs["top3"] * batch_size
            running_top5 += accs["top5"] * batch_size
            n_samples += batch_size

    epoch_loss = running_loss / n_samples
    epoch_top1 = running_top1 / n_samples
    epoch_top3 = running_top3 / n_samples
    epoch_top5 = running_top5 / n_samples

    metrics = {
        "loss": epoch_loss,
        "top1": epoch_top1,
        "top3": epoch_top3,
        "top5": epoch_top5}
    return metrics



@torch.no_grad()
def eval_one_epoch(model: nn.Module,
                   dataloader,
                   device: torch.device,
                   criterion: nn.Module):
    """
    Evalúa una época en el loader de validación / test.
    Devuelve: dict con loss, top1, top3, top5.
    """
    model.eval()

    running_loss = 0.0
    running_top1 = 0.0
    running_top3 = 0.0
    running_top5 = 0.0
    n_samples = 0

    for imgs, labels in dataloader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        batch_size = imgs.size(0)

        outputs = model(imgs)
        loss = criterion(outputs, labels)

        accs = accuracy_topk(outputs, labels, topk=(1, 3, 5))

        running_loss += loss.item() * batch_size
        running_top1 += accs["top1"] * batch_size
        running_top3 += accs["top3"] * batch_size
        running_top5 += accs["top5"] * batch_size
        n_samples += batch_size

    epoch_loss = running_loss / n_samples
    epoch_top1 = running_top1 / n_samples
    epoch_top3 = running_top3 / n_samples
    epoch_top5 = running_top5 / n_samples

    metrics = {
        "loss": epoch_loss,
        "top1": epoch_top1,
        "top3": epoch_top3,
        "top5": epoch_top5}
    return metrics
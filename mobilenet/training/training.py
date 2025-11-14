from collections import deque
import torch
import torch.nn as nn

from training.diff_augment import *
from training.one_epoch_loop import *


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_mobilenet(
    model: nn.Module,
    train_loader,
    val_loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    criterion: nn.Module,
    diff_augment: nn.Module = None,
    num_epochs: int = 50,
    patience: int = 5,
    early_stop_delta: float = 1e-3,
    acc_targets=None,
    verbose: bool = True,
):
    """
    Entrena MobileNet con:
      - early stopping por plateau de loss en validación
      - early stopping por alcanzar thresholds de top-k accuracy.

    Parámetros:
      - model: MobileNet v1/v2
      - train_loader, val_loader
      - optimizer
      - device
      - criterion: CrossEntropyLoss, etc.
      - diff_augment: instancia de DiffAugment o None
      - num_epochs: máximo de épocas
      - patience: número de épocas para mirar el plateau del loss
      - early_stop_delta: tolerancia de variación en loss para early stopping
      - acc_targets:
           None → no usa este criterio
           float → target para top1
           (t1, t3) → targets para top1, top3
           (t1, t3, t5) → targets para top1, top3, top5
      - verbose: imprime métricas por época

    Retorna:
      - model (entrenado)
      - history: dict con listas por época (train_loss, val_loss, etc.)
    """

    model.to(device)

    # Normalizar acc_targets
    top_keys = ["top1", "top3", "top5"]
    if acc_targets is None:
        acc_targets_norm = None
    elif isinstance(acc_targets, (float, int)):
        acc_targets_norm = (float(acc_targets),)
    else:
        acc_targets_norm = tuple(float(x) for x in acc_targets)

    # Historial
    history = {"train_loss": [],
        "train_top1": [],
        "train_top3": [],
        "train_top5": [],
        "val_loss": [],
        "val_top1": [],
        "val_top3": [],
        "val_top5": [],}

    val_loss_window = deque(maxlen=patience)

    for epoch in range(1, num_epochs + 1):

        # Training one epoc
        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            criterion=criterion,
            diff_augment=diff_augment)

        # Validatiom
        val_metrics = eval_one_epoch(model=model,
            dataloader=val_loader,
            device=device,
            criterion=criterion,)

        # Guardar en historial
        history["train_loss"].append(train_metrics["loss"])
        history["train_top1"].append(train_metrics["top1"])
        history["train_top3"].append(train_metrics["top3"])
        history["train_top5"].append(train_metrics["top5"])

        history["val_loss"].append(val_metrics["loss"])
        history["val_top1"].append(val_metrics["top1"])
        history["val_top3"].append(val_metrics["top3"])
        history["val_top5"].append(val_metrics["top5"])

        if verbose:
            print(f"Epoch [{epoch}/{num_epochs}]")
            print(f"  Train: loss={train_metrics['loss']:.4f}, "
                  f"top1={train_metrics['top1']*100:.2f}%, "
                  f"top3={train_metrics['top3']*100:.2f}%, "
                  f"top5={train_metrics['top5']*100:.2f}%")
            print(f"  Val  : loss={val_metrics['loss']:.4f}, "
                  f"top1={val_metrics['top1']*100:.2f}%, "
                  f"top3={val_metrics['top3']*100:.2f}%, "
                  f"top5={val_metrics['top5']*100:.2f}%")

        # Early stopping por accuracy (top-k)
        stopped_by_acc = False
        if acc_targets_norm is not None:
            conds = []
            for thr, key in zip(acc_targets_norm, top_keys):
                conds.append(val_metrics[key] >= thr)
            if all(conds):
                if verbose:
                    used_keys = ", ".join(
                        [f"{k}>={thr:.3f}" for thr, k in zip(acc_targets_norm, top_keys)])
                    
                    print(f">>> Early stopping por accuracy (condiciones: {used_keys})")
                stopped_by_acc = True

        if stopped_by_acc:
            break

        # Early stopping por plateau de loss
        val_loss_window.append(val_metrics["loss"])

        if len(val_loss_window) == patience:
            max_loss = max(val_loss_window)
            min_loss = min(val_loss_window)
            if (max_loss - min_loss) <= early_stop_delta:
                if verbose:
                    print(f">>> Early stopping por plateau de loss "
                          f"(últimas {patience} épocas, Δ={max_loss - min_loss:.6f} ≤ {early_stop_delta})")
                break

    return model, history
from typing import List, Dict, Tuple
import numpy as np
import torch

@torch.inference_mode()
def topk_accuracy_on_loader(model,
    dataloader,
device,topk = (1, 3, 5),) :
    """
    Calcula accuracy@k para un modelo multiclase en un DataLoader.

    Parameters
    ----------
    model : torch.nn.Module
        Modelo entrenado (en modo eval).
    dataloader : DataLoader
        Conjunto de evaluación (val/test).
    device : torch.device
        cpu o cuda.
    topk : tuple of int
        Valores de k que se quieren evaluar (ej. (1,3,5)).

    Returns
    -------
    Dict[int, float]
        Accuracy@k promedio (entre 0 y 1) para cada k.
    """
    model.eval()
    correct_counts = {k: 0 for k in topk}
    total = 0

    for xb, yb in dataloader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        logits = model(xb)
        _, pred_topk = logits.topk(max(topk), dim=1, largest=True, sorted=True)
        total += yb.size(0)

        for k in topk:
            correct_k = pred_topk[:, :k].eq(yb.view(-1, 1)).any(dim=1).sum().item()
            correct_counts[k] += correct_k

    return {k: correct_counts[k] / total for k in topk}


def print_topk_report(accs: Dict[int, float], digits: int = 3):
    """Imprime Top-k accuracies formateadas."""
    print("=== Top-k Accuracy ===")
    for k, v in accs.items():
        print(f"Top-{k:<2d} : {v * 100:.{digits}f}%")
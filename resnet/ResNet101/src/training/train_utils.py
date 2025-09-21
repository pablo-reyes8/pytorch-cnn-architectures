import torch

@torch.no_grad()
def accuracy_top1(logits, targets) :
    """Accuracy (%) para clasificación con 2 clases (o multi-clase), usando argmax."""
    preds = logits.argmax(dim=1)
    correct = (preds == targets).sum().item()
    return 100.0 * correct / max(1, targets.size(0))


def _update_binary_counts(preds, y):
    """
    Retorna (tp, fp, fn, tn) as ints para problema binario con clases {0,1}.
    preds y y son tensores 1D de enteros.
    """
    tp = ((preds == 1) & (y == 1)).sum().item()
    fp = ((preds == 1) & (y == 0)).sum().item()
    fn = ((preds == 0) & (y == 1)).sum().item()
    tn = ((preds == 0) & (y == 0)).sum().item()
    return tp, fp, fn, tn


def _safe_prec_recall_f1(tp, fp, fn):
    """Devuelve (precision, recall, f1) en [0,1] con divisiones seguras."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1
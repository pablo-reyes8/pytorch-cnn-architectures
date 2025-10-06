import torch
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
import random

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parámetros: {total_params:,}")
    print(f"Parámetros entrenables: {trainable_params:,}")
    return total_params, trainable_params


def summary_params(model):
    print("="*60)
    print("Resumen de parámetros por capa")
    print("="*60)
    for name, p in model.named_parameters():
        print(f"{name:50s} | shape={tuple(p.shape)} | params={p.numel():,}")
    total = sum(p.numel() for p in model.parameters())
    print("="*60)
    print(f"Total parámetros: {total:,}")


@torch.no_grad()
def evaluate_top1(model, loader, device="cuda"):
    model.eval()                 
    device = torch.device(device)
    model.to(device)

    correct, total = 0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        logits = model(X)   
        preds = logits.argmax(1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    return correct / total

@torch.no_grad()
def eval_report(model, loader, device="cuda", class_names=None):
    model.eval()
    y_true, y_pred = [], []

    for X, y in loader:
        X = X.to(device)
        logits = model(X)
        y_true.append(y.numpy())
        y_pred.append(logits.argmax(1).cpu().numpy())

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    plt.xlabel("Predicción")
    plt.ylabel("Etiqueta real")
    plt.title("Matriz de confusión")
    plt.show()

    print(classification_report(y_true, y_pred, target_names=class_names))

@torch.no_grad()
def plot_multiclass_roc(
    model,
    loader,
    num_classes: int,
    device: str = "cuda",
    class_names=None,
    plot_per_class: bool = True):
    """
    Grafica ROC para problema multiclase (one-vs-rest) + micro y macro promedio.

    - model: nn.Module (en eval() devuelve logits [B,C])
    - loader: DataLoader de evaluación
    - num_classes: número de clases C
    - class_names: lista opcional de nombres (len=C)
    - plot_per_class: si True, dibuja curvas por clase; si False, solo micro/macro
    """
    device = torch.device(device)
    model.to(device)
    model.eval()

    all_probs = []
    all_labels = []
    for X, y in loader:
        X = X.to(device)
        logits = model(X)                # [B,C]
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(y.cpu().numpy())

    y_true = np.concatenate(all_labels, axis=0)               # [N]
    y_score = np.concatenate(all_probs, axis=0)               # [N,C]

    y_true_bin = label_binarize(y_true, classes=np.arange(num_classes))  # [N,C]

    # 3) ROC por clase
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for c in range(num_classes):
        fpr[c], tpr[c], _ = roc_curve(y_true_bin[:, c], y_score[:, c])
        roc_auc[c] = auc(fpr[c], tpr[c])

    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    all_fpr = np.unique(np.concatenate([fpr[c] for c in range(num_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for c in range(num_classes):
        mean_tpr += np.interp(all_fpr, fpr[c], tpr[c])
    mean_tpr /= num_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])


    plt.figure(figsize=(8, 6))
    plt.plot(fpr["micro"], tpr["micro"],
             linestyle='-', linewidth=2,
             label=f"micro-avg (AUC = {roc_auc['micro']:.3f})")
    plt.plot(fpr["macro"], tpr["macro"],
             linestyle='--', linewidth=2,
             label=f"macro-avg (AUC = {roc_auc['macro']:.3f})")

    if plot_per_class:
        for c in range(num_classes):
            name = class_names[c] if class_names else f"Class {c}"
            plt.plot(fpr[c], tpr[c], linewidth=1,
                     label=f"{name} (AUC = {roc_auc[c]:.3f})")
    plt.plot([0, 1], [0, 1], linestyle=':', linewidth=1)

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC multiclase (one-vs-rest)")
    plt.legend(loc="lower right", fontsize=9, ncol=1 if plot_per_class else 2)
    plt.tight_layout()
    plt.show()

    return roc_auc


@torch.no_grad()
def show_batch_predictions(model, loader, class_names, device="cuda", n=8):
    """
    Muestra un batch de imágenes con etiquetas reales y predicciones del modelo.
    
    - model: red entrenada (GoogLeNet, etc.)
    - loader: DataLoader (train o test)
    - class_names: lista con nombres de las clases
    - device: "cuda" o "cpu"
    - n: número de imágenes a mostrar
    """
    model.eval()
    device = torch.device(device)
    model.to(device)

    X, y = next(iter(loader))  # toma un batch
    X, y = X.to(device), y.to(device)

    logits = model(X)
    preds = logits.argmax(1)

    X = X.cpu().permute(0, 2, 3, 1).numpy()  # a formato (H,W,C)
    fig, axes = plt.subplots(1, n, figsize=(3*n, 3))
    
    for i in range(n):
        img = X[i]
        img = (img - img.min()) / (img.max() - img.min())  # normaliza 0-1 para mostrar
        true_label = class_names[y[i].item()]
        pred_label = class_names[preds[i].item()]
        
        axes[i].imshow(img)
        axes[i].axis("off")
        axes[i].set_title(f"T:{true_label}\nP:{pred_label}",
                          fontsize=10,
                          color=("green" if true_label==pred_label else "red"))
    
    plt.tight_layout()
    plt.show()



def _norm01(x):
    x = x.astype(np.float32)
    mn, mx = x.min(), x.max()
    if mx > mn:
        x = (x - mn) / (mx - mn)
    else:
        x = np.zeros_like(x)
    return x

@torch.no_grad()
def plot_inception_v1_block_activations(
    model,
    loader,
    block_attr='in3a',    # nombre del bloque: in3a, in3b, etc.
    n_filters=10,
    device='cuda'):
    """
    Muestra mapas de activación PRE (Conv2d) y POST (ConvBNReLU) para las ramas:
    - 1x1:        branch1
    - 3x3:        branch2_conv   (tras branch2_reduce)
    - 5x5:        branch3_conv   (tras branch3_reduce)
    - pool->1x1:  branch4_proj   (tras branch4_pool)

    Crea dos figuras; cada una es una rejilla 4 x n_filters (filas=ramas, cols=filtros).
    """
    model.eval()
    device = torch.device(device if (device == 'cpu' or torch.cuda.is_available()) else 'cpu')
    model.to(device)

    rand_idx = random.randrange(len(loader))
    for i, (Xb, yb) in enumerate(loader):
        if i == rand_idx:
            x = Xb[0:1].to(device)   # una imagen
            break

    try:
        block = getattr(model, block_attr)
    except AttributeError as e:
        raise ValueError(f"No existe '{block_attr}' en el modelo. Revisa el nombre del bloque.") from e

    branches_post = {
        '1x1': block.branch1,           # ConvBNReLU
        '3x3': block.branch2_conv,      # ConvBNReLU final (después de reduce)
        '5x5': block.branch3_conv,      # ConvBNReLU final
        'pool1x1': block.branch4_proj,  # ConvBNReLU después del MaxPool
    }
    branch_order = ['1x1', '3x3', '5x5', 'pool1x1']

    acts_pre = {}
    acts_post = {}
    hooks = []

    def make_pre_hook(name):
        def hook(module, inp, out):
            acts_pre[name] = out.detach().cpu()
        return hook

    def make_post_hook(name):
        def hook(module, inp, out):
            acts_post[name] = out.detach().cpu()
        return hook

    # Registrar hooks por rama
    for name in branch_order:
        mod_post = branches_post[name]          # ConvBNReLU (tiene .net = Sequential[Conv2d, BN, ReLU])
        if not hasattr(mod_post, 'net') or len(mod_post.net) < 3:
            raise RuntimeError(f"La rama '{name}' no parece ser ConvBNReLU(net[Conv, BN, ReLU]).")

        conv2d = mod_post.net[0]              
        hooks.append(conv2d.register_forward_hook(make_pre_hook(name)))
        hooks.append(mod_post.register_forward_hook(make_post_hook(name)))  

    _ = model(x)

    for h in hooks:
        h.remove()

    def _plot_mode(acts, title):
        fig, axes = plt.subplots(len(branch_order), n_filters,
                                 figsize=(n_filters*2.2, len(branch_order)*2.2))
        if len(branch_order) == 1 or n_filters == 1:
            axes = np.array(axes).reshape(len(branch_order), n_filters)

        for r, name in enumerate(branch_order):
            A = acts[name][0]   
            C = A.shape[0]
            use = min(C, n_filters)
            for c in range(use):
                fmap = A[c].numpy()
                fmap = _norm01(fmap)
                axes[r, c].imshow(fmap, cmap='viridis', interpolation='nearest')
                axes[r, c].axis('off')
                if r == 0:
                    axes[r, c].set_title(f"F{c}", fontsize=9)

            for c in range(use, n_filters):
                axes[r, c].axis('off')

            axes[r, 0].set_ylabel(name, rotation=0, ha='right', va='center',
                                  fontsize=10, labelpad=25)

        plt.suptitle(f"{block_attr} — {title}", y=1.02, fontsize=12)
        plt.tight_layout()
        plt.show()


    _plot_mode(acts_pre, "PRE (salida Conv2d, antes de BN/ReLU)")
    _plot_mode(acts_post, "POST (salida ConvBNReLU, después de ReLU)")
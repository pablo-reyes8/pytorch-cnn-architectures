import torch
import matplotlib.pyplot as plt
from torch.nn import Module

def show_first_layer_filters(model, max_filters=32):
    first_conv = None
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            first_conv = m
            break

    if first_conv is None:
        raise ValueError("No se encontró capa Conv2d en el modelo")

    weights = first_conv.weight.data.clone().cpu() 
    n_filters = min(max_filters, weights.shape[0])
    plt.figure(figsize=(12, 6))
    for i in range(n_filters):
        f = weights[i]
        f = (f - f.min()) / (f.max() - f.min() + 1e-5)  
        f = f.permute(1, 2, 0) 
        plt.subplot(4, n_filters//4, i+1)
        plt.imshow(f)
        plt.axis("off")
    plt.suptitle("Filtros de la primera capa conv")
    plt.show()


def show_feature_maps(model, layer_name, image, device="cuda", max_maps=16):
    """
    Visualiza activaciones (feature maps) de una capa específica.
    - layer_name: str con nombre de la capa (ej. 'features.0.block.0' si usas Sequential)
    - image: tensor [1,C,H,W] normalizado
    """
    activations = {}

    def hook_fn(module, inp, out):
        activations["feat"] = out.detach().cpu()

    layer = dict([*model.named_modules()])[layer_name]
    hook = layer.register_forward_hook(hook_fn)

    model.eval()
    _ = model(image.to(device))  

    hook.remove()
    fmap = activations["feat"] 

    n_maps = min(max_maps, fmap.shape[1])
    plt.figure(figsize=(12, 6))
    for i in range(n_maps):
        plt.subplot(4, n_maps//4, i+1)
        fm = fmap[0, i].numpy()
        fm = (fm - fm.min()) / (fm.max() - fm.min() + 1e-5)
        plt.imshow(fm, cmap="viridis")
        plt.axis("off")
    plt.suptitle(f"Feature maps de {layer_name}")
    plt.show()


def plot_weight_histograms(model):
    plt.figure(figsize=(12,6))
    i = 1
    for name, param in model.named_parameters():
        if "weight" in name:
            plt.subplot(3,4,i)
            plt.hist(param.detach().cpu().numpy().ravel(), bins=40)
            plt.title(name)
            i += 1
            if i > 12: break
    plt.tight_layout()
    plt.show()


def plot_training_history(history):
    """
    history: dict con 'train_acc', 'val_acc', 'train_f1', 'val_f1' (como guardaste en tu loop)
    """
    plt.figure(figsize=(14,5))

    # Accuracy
    plt.subplot(1,2,1)
    plt.plot(history.get("train_acc", []), label="Train Acc")
    plt.plot(history.get("val_acc", []), label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy durante el entrenamiento")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # F1
    plt.subplot(1,2,2)
    plt.plot(history.get("train_f1", []), label="Train F1")
    plt.plot(history.get("val_f1", []), label="Val F1")
    plt.xlabel("Epoch")
    plt.ylabel("F1-score")
    plt.title("F1 durante el entrenamiento")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def plot_fc_activations(model, val_loader, device="cuda", layer_name="classifier.0"):
    """
    layer_name: nombre de la capa en model.named_modules()
    """
    activations = {}

    def hook_fn(module, inp, out):
        activations["feat"] = out.detach().cpu()

    layer = dict([*model.named_modules()])[layer_name]
    hook = layer.register_forward_hook(hook_fn)

    xb, _ = next(iter(val_loader))
    xb = xb.to(device)
    model.eval()
    _ = model(xb)

    hook.remove()
    feats = activations["feat"]  # [B, D]

    plt.figure(figsize=(10,4))
    plt.hist(feats.numpy().ravel(), bins=50, alpha=0.7, color="purple")
    plt.title(f"Distribución de activaciones en {layer_name}")
    plt.xlabel("Valor de activación")
    plt.ylabel("Frecuencia")
    plt.show()


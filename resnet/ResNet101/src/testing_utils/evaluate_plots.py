import torch
import numpy as np
from sklearn.metrics import classification_report, roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import torch.nn.functional as F
import math

from src.testing_utils.test_utils import *



def show_layer_filters(model, layer_name, max_filters=32, mode="mean"):
    """
    mode:
      - "mean": promedia canales de entrada -> 2D (recomendado)
      - "absmax": canal con mayor norma L2 (elige el más 'fuerte')
      - "first3": si hay >=3 canales, usa los primeros 3 como pseudo-RGB
    """
    layer = dict(model.named_modules()).get(layer_name, None)
    if layer is None or not isinstance(layer, torch.nn.Conv2d):
        raise ValueError(f"No se encontró una Conv2d llamada '{layer_name}'")

    W = layer.weight.detach().cpu()
    out_c, in_c, kH, kW = W.shape
    n = min(max_filters, out_c)

    cols = min(8, n)
    rows = math.ceil(n / cols)
    plt.figure(figsize=(1.8*cols, 1.8*rows))

    for i in range(n):
        f = W[i]
        if in_c == 3 and mode == "first3":
            img = f.permute(1, 2, 0)
        elif mode == "absmax" and in_c > 1:
            ch = torch.argmax(f.flatten(1).pow(2).sum(dim=1)).item()
            img = f[ch]
        elif mode == "first3" and in_c >= 3:
            img = f[:3].permute(1, 2, 0)
        else:
            img = f.mean(dim=0)

        # normalizar a 0-1
        mn, mx = img.min(), img.max()
        img = (img - mn) / (mx - mn + 1e-5)

        plt.subplot(rows, cols, i+1)
        if img.ndim == 2:
            plt.imshow(img, cmap="gray")
        else:
            plt.imshow(img)  # asume 3 canales
        plt.axis("off")

    plt.suptitle(f"Filtros de '{layer_name}'  (mode={mode})", y=0.98)
    plt.tight_layout()
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


def plot_fc_activations(model, val_loader, device="cuda", layer_name="layer4.2.conv2"):
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


def show_batch_images(images, labels, preds=None, probs=None, n=16, title=None,
                      classes=None, mean=[0.48293063044548035, 0.44492557644844055, 0.3957090973854065], 
                      std=[0.2592383325099945, 0.25327032804489136, 0.2598187029361725], ncols=8):
    """
    Muestra un grid de imágenes con sus etiquetas y (opcional) predicciones.

    Args:
        images: Tensor [N,C,H,W] o lista de tensores [C,H,W].
        labels: lista/array/tensor de ints (clases verdaderas)
        preds : lista/array/tensor de ints (predicciones opcionales)
        probs : lista/array/tensor de floats (confianza opcional por predicción top-1)
        n     : cuántas imágenes mostrar
        classes: lista de nombres de clases (opcional, para mostrar texto legible)
        mean,std: usados para desnormalizar (si aplicaste Normalize)
        ncols : columnas del grid
    """
    # Asegurar tipos
    if torch.is_tensor(images):
        images = [images[i] for i in range(min(n, images.size(0)))]
    else:
        images = images[:n]
    n = len(images)
    labels = labels[:n]
    preds  = preds[:n] if preds is not None else None
    probs  = probs[:n] if probs is not None else None

    # Desnormaliza cada imagen (si fue normalizada)
    images_denorm = [denormalize(img, mean=mean, std=std) for img in images]

    nrows = (n + ncols - 1) // ncols
    plt.figure(figsize=(2.4*ncols, 2.7*nrows))
    for i in range(n):
        plt.subplot(nrows, ncols, i+1)
        npimg = _to_numpy_img(images_denorm[i])

        if npimg.ndim == 2:
            plt.imshow(npimg, cmap='gray')
        else:
            plt.imshow(npimg)

        # Construir título
        y = int(labels[i])
        ytxt = classes[y] if (classes is not None and 0 <= y < len(classes)) else str(y)

        if preds is None:
            title_i = f"y={ytxt}"
            color = 'black'
        else:
            p = int(preds[i])
            ptxt = classes[p] if (classes is not None and 0 <= p < len(classes)) else str(p)
            ok = (p == y)
            conf_txt = ""
            if probs is not None:
                conf = float(probs[i])
                conf_txt = f" ({conf:.2f})"
            title_i = f"y={ytxt} | ŷ={ptxt}{conf_txt}"
            color = 'green' if ok else 'red'

        plt.title(title_i, color=color, fontsize=10)
        plt.axis('off')

    if title:
        plt.suptitle(title, y=1.02, weight='bold')
    plt.tight_layout()
    plt.show()


@torch.no_grad()
def visualize_test_predictions(model, test_loader, device='cpu', n=16, only_errors=False,
                               classes=None, mean=[0.48293063044548035, 0.44492557644844055, 0.3957090973854065], 
                               std=[0.2592383325099945, 0.25327032804489136, 0.2598187029361725],
                               return_data=False, with_confidence=True):
    """
    Toma batches del test_loader, corre el modelo y muestra un grid con:
    etiqueta real, predicción y (opcional) confianza de la predicción top-1.

    Args:
        only_errors: si True, muestra solo mal clasificadas
        with_confidence: si True, muestra probabilidad softmax de la predicción
        return_data: si True, retorna (imgs, labels, preds, probs) además de visualizar
    """
    if isinstance(device, str):
        device = torch.device(device)

    model.eval()
    imgs, labs, preds, confs = [], [], [], []

    for xb, yb in test_loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        logits = model(xb)
        pred = torch.argmax(logits, 1)

        if only_errors:
            mask = pred.ne(yb)
        else:
            mask = torch.ones_like(yb, dtype=torch.bool)

        if with_confidence:
            prob = F.softmax(logits, dim=1).max(dim=1).values  # top-1 confidence

        sel = mask.nonzero(as_tuple=False).squeeze(1)
        for idx in sel:
            imgs.append(xb[idx].cpu())
            labs.append(int(yb[idx].cpu()))
            preds.append(int(pred[idx].cpu()))
            if with_confidence:
                confs.append(float(prob[idx].cpu()))
            if len(imgs) >= n:
                break
        if len(imgs) >= n:
            break

    if len(imgs) == 0:
        print("No hay ejemplos que cumplan el criterio (quizá el modelo acertó todo ese batch).")
        return None if not return_data else ([], [], [], [])

    title = "Test samples (pred vs true)" if not only_errors else "Errores del modelo en test"
    show_batch_images(imgs, labs, preds, confs if with_confidence else None, n=n, title=title,
                      classes=classes, mean=mean, std=std)

    if return_data:
        return imgs, labs, preds, (confs if with_confidence else None)
    



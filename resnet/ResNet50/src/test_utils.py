import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
from typing import Dict, List, Iterable, Optional
import torchvision

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)

def denormalize(img_tensor, mean=CIFAR10_MEAN, std=CIFAR10_STD):
    """
    Invierte Normalize: x * std + mean
    Soporta [C,H,W] ó [N,C,H,W].
    """
    if img_tensor.dim() == 4:
        C = img_tensor.size(1)
    else:
        C = img_tensor.size(0)
    mean_t = torch.tensor(mean[:C], device=img_tensor.device).view(1 if img_tensor.dim()==4 else C, 1, 1)
    std_t  = torch.tensor(std[:C],  device=img_tensor.device).view(1 if img_tensor.dim()==4 else C, 1, 1)
    return img_tensor * std_t + mean_t

def _to_numpy_img(img_chw):
    """
    Convierte un tensor [C,H,W] en imagen numpy HxWxC (RGB o 1 canal).
    Clampa a [0,1] para mostrar.
    """
    img = img_chw.detach().cpu().clamp(0, 1)
    if img.size(0) == 1:
        return img.squeeze(0).numpy()
    else:
        return img.permute(1, 2, 0).numpy()

def show_batch_images(images, labels, preds=None, probs=None, n=16, title=None,
                      classes=None, mean=CIFAR10_MEAN, std=CIFAR10_STD, ncols=8):
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
                               classes=None, mean=CIFAR10_MEAN, std=CIFAR10_STD,
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
    


def _get_module_by_name(model: torch.nn.Module, name: str) -> torch.nn.Module:
    """
    Permite obtener un sub-módulo vía 'ruta' (p.ej. 'layer3.0.conv2').
    """
    mod = model
    for attr in name.split('.'):
        if attr.isdigit():   # para secuenciales: "layer3.0"
            mod = mod[int(attr)]
        else:
            mod = getattr(mod, attr)
    return mod

def _make_grid_chw(t: torch.Tensor, nrow: int = 8):
    """
    t: [N, C, H, W] (0..1) -> grid numpy HxWxC.
    """
    grid = torchvision.utils.make_grid(t.cpu(), nrow=nrow, padding=1)
    npimg = grid.permute(1, 2, 0).numpy()
    return npimg

# =========================================================
# (a) ACTIVACIONES INTERMEDIAS
# ========================================================='

class ActivationCatcher:
    """
    Registra forward hooks en capas (por nombre) y guarda las activaciones.
    Uso:
        catcher = ActivationCatcher(model, ["layer2.0.conv2", "layer3.0.conv2"])
        with catcher.capture():
            _ = model(xb)
        acts = catcher.activations  # dict[name] -> tensor [N,C,H,W]
    """
    def __init__(self, model: torch.nn.Module, layer_names: Iterable[str]):
        self.model = model
        self.layer_names = list(layer_names)
        self.handles = []
        self.activations: Dict[str, torch.Tensor] = {}

    def _hook(self, name):
        def fn(_, __, output):
            self.activations[name] = output.detach()
        return fn

    def capture(self):
        # context manager
        class _Ctx:
            def __init__(self, outer):
                self.outer = outer
            def __enter__(self):
                for name in self.outer.layer_names:
                    m = _get_module_by_name(self.outer.model, name)
                    h = m.register_forward_hook(self.outer._hook(name))
                    self.outer.handles.append(h)
                self.outer.activations.clear()
                return self.outer
            def __exit__(self, exc_type, exc, tb):
                for h in self.outer.handles:
                    h.remove()
                self.outer.handles.clear()
        return _Ctx(self)

def show_feature_maps(acts: torch.Tensor, max_maps: int = 16, ncols: int = 8, title: Optional[str] = None):
    """
    Visualiza hasta max_maps feature maps de un tensor de activaciones [N,C,H,W].
    Usa el primer elemento del batch (N=0).
    """
    assert acts.dim() == 4, "Se espera [N, C, H, W]"
    # elegimos la muestra 0 y los primeros canales
    a = acts[0]                        # [C,H,W]
    C = min(max_maps, a.size(0))
    a = a[:C]                          # [C,H,W]

    # Normaliza canal a canal para visualización (0..1)
    a = a.clone()
    a = (a - a.amin(dim=(1,2), keepdim=True)) / (a.amax(dim=(1,2), keepdim=True) - a.amin(dim=(1,2), keepdim=True) + 1e-8)

    grid_np = _make_grid_chw(a.unsqueeze(1), nrow=ncols)   # [C,1,H,W] -> grid gris
    plt.figure(figsize=(2.2*ncols, 2.2*((C+ncols-1)//ncols)))
    plt.imshow(grid_np.squeeze(), cmap='viridis')
    plt.axis('off')
    if title:
        plt.title(title)
    plt.show()


# =========================================================
# (b) FILTROS APRENDIDOS (PRIMERA CONV)
# =========================================================

def show_first_conv_filters(model: torch.nn.Module, layer_name: str = "conv1",
                            max_filters: int = 64, ncols: int = 8):
    """
    Visualiza kernels de la primera conv (p.ej. 'conv1' o 'layer2.0.conv1' si quieres).
    Para entrada RGB, se muestran como imágenes RGB (normalizando por canal).
    """
    conv = _get_module_by_name(model, layer_name)
    assert isinstance(conv, torch.nn.Conv2d), f"{layer_name} no es Conv2d"
    w = conv.weight.detach().cpu()     # [out, in, kh, kw]
    out_ch, in_ch, kh, kw = w.shape

    F = min(max_filters, out_ch)
    w = w[:F]

    # Normalizamos cada filtro para visualizar
    w = w.clone()
    w_flat = w.view(F, -1)
    w = (w - w_flat.min(dim=1, keepdim=True)[0].view(F,1,1,1)) / \
        (w_flat.max(dim=1, keepdim=True)[0].view(F,1,1,1) - w_flat.min(dim=1, keepdim=True)[0].view(F,1,1,1) + 1e-8)

    if in_ch == 3:
        # mostrar como RGB
        grid_np = _make_grid_chw(w, nrow=ncols)
        plt.figure(figsize=(2.2*ncols, 2.2*((F+ncols-1)//ncols)))
        plt.imshow(grid_np)
        plt.axis('off')
        plt.title(f"Filtros {layer_name} (RGB) [{F}/{out_ch}]")
        plt.show()
    else:
        # mostrar cada canal como mapa de calor gris (stackeamos canales como 'batch')
        wg = w[:, :1]  # solo primer canal para visualizar; o podrías promediar canales
        grid_np = _make_grid_chw(wg, nrow=ncols)
        plt.figure(figsize=(2.2*ncols, 2.2*((F+ncols-1)//ncols)))
        plt.imshow(grid_np.squeeze(), cmap='viridis')
        plt.axis('off')
        plt.title(f"Filtros {layer_name} (ch0) [{F}/{out_ch}]")
        plt.show()



# =========================================================
# (c) HISTOGRAMAS (PESOS Y ACTIVACIONES)
# =========================================================
def plot_weight_histograms(model: torch.nn.Module,
                           module_names: Optional[List[str]] = None,
                           bins: int = 80, title: str = "Distribución de pesos"):
    """
    Histogramas de pesos para capas seleccionadas (por nombre). Si no pasas nombres, recorre todas las Conv/Linear.
    """
    params = []
    labels = []

    if module_names is None:
        # por defecto: todas las conv/linear
        for name, m in model.named_modules():
            if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
                params.append(m.weight.detach().flatten().cpu())
                labels.append(name)
    else:
        for name in module_names:
            m = _get_module_by_name(model, name)
            assert isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)), f"{name} no es Conv/Linear"
            params.append(m.weight.detach().flatten().cpu())
            labels.append(name)

    n = len(params)
    cols = 3
    rows = (n + cols - 1) // cols
    plt.figure(figsize=(5*cols, 3.5*rows))
    for i, (p, lab) in enumerate(zip(params, labels), 1):
        plt.subplot(rows, cols, i)
        plt.hist(p.numpy(), bins=bins, alpha=0.8)
        plt.title(lab)
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

@torch.no_grad()
def plot_activation_histograms(model: torch.nn.Module, xb: torch.Tensor,
                               layer_names: List[str], bins: int = 80,
                               title: str = "Distribución de activaciones (post-BN/ReLU)"):
    """
    Pasa un batch por el modelo y hace histogramas de activaciones de capas elegidas.
    layer_names: nombres de módulos sobre los que enganchar el hook (típicamente convs dentro de bloques).
    """
    catcher = ActivationCatcher(model, layer_names)
    model.eval()
    with catcher.capture():
        _ = model(xb)

    acts = catcher.activations
    n = len(layer_names)
    cols = 3
    rows = (n + cols - 1) // cols
    plt.figure(figsize=(5*cols, 3.5*rows))
    for i, name in enumerate(layer_names, 1):
        a = acts[name].detach().flatten().cpu().numpy()
        plt.subplot(rows, cols, i)
        plt.hist(a, bins=bins, alpha=0.8)
        plt.title(name)
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


import numpy as np

def compute_confusion_matrix(dataloader, model, device, num_classes: int):
    model.eval()
    cm = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    for xb, yb in dataloader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        logits = model(xb)
        preds = logits.argmax(1)
        for t, p in zip(yb.view(-1), preds.view(-1)):
            cm[t.long(), p.long()] += 1
    return cm.cpu().numpy()

def plot_confusion_matrix(cm: np.ndarray, class_names=None, normalize='true', title="Confusion Matrix"):
    """
    normalize: None | 'true' (por fila=recall) | 'pred' (por columna=precision) | 'all'
    """
    cm_plot = cm.astype(float)
    if normalize == 'true':
        cm_plot = cm_plot / (cm_plot.sum(axis=1, keepdims=True) + 1e-12)
    elif normalize == 'pred':
        cm_plot = cm_plot / (cm_plot.sum(axis=0, keepdims=True) + 1e-12)
    elif normalize == 'all':
        cm_plot = cm_plot / (cm_plot.sum() + 1e-12)

    plt.figure(figsize=(7, 6))
    im = plt.imshow(cm_plot, interpolation='nearest', cmap='Blues')
    plt.title(title)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    tick_marks = np.arange(cm.shape[0])
    if class_names is None:
        class_names = [str(i) for i in tick_marks]
    plt.xticks(tick_marks, class_names, rotation=45, ha='right')
    plt.yticks(tick_marks, class_names)

    thresh = cm_plot.max() * 0.6
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            txt = f"{cm_plot[i, j]:.2f}" if normalize else str(cm[i, j])
            plt.text(j, i, txt, ha="center", va="center",
                     color="white" if cm_plot[i, j] > thresh else "black", fontsize=8)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()

def per_class_accuracy(cm: np.ndarray):
    # recall por clase = diag / suma fila
    recalls = (np.diag(cm) / (cm.sum(axis=1) + 1e-12)) * 100.0
    return recalls


import cv2

def denormalize_img(x_chw, mean=CIFAR10_MEAN, std=CIFAR10_STD):
    C = x_chw.size(0)
    mean_t = torch.tensor(mean[:C]).view(C,1,1)
    std_t  = torch.tensor(std[:C]).view(C,1,1)
    return (x_chw.cpu() * std_t + mean_t).clamp(0,1)

def to_numpy_rgb(x_chw):
    x = x_chw.permute(1,2,0).numpy()  # HWC
    return (x * 255).astype(np.uint8)

def overlay_cam_on_image(img_rgb, cam, alpha=0.35):
    heatmap = cv2.applyColorMap((cam*255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = (alpha*heatmap + (1-alpha)*img_rgb).astype(np.uint8)
    return overlay

class GradCAM:
    """
    Grad-CAM genérico para cualquier capa conv referida por nombre (e.g. 'layer5.2.conv3').
    """
    def __init__(self, model, target_layer_name: str):
        self.model = model
        self.model.eval()
        self.target_layer = self._get_module_by_name(target_layer_name)

        self.activations = None
        self.gradients = None
        self.fwd_handle = self.target_layer.register_forward_hook(self._fwd_hook)
        self.bwd_handle = self.target_layer.register_full_backward_hook(self._bwd_hook)

    def _get_module_by_name(self, name: str):
        m = self.model
        for attr in name.split('.'):
            if attr.isdigit():
                m = m[int(attr)]
            else:
                m = getattr(m, attr)
        return m

    def _fwd_hook(self, module, inp, out):
        self.activations = out.detach()           # [B, C, H, W]

    def _bwd_hook(self, module, grad_input, grad_output):
        # grad_output is a tuple with one element: grad wrt output
        self.gradients = grad_output[0].detach()  # [B, C, H, W]

    def __call__(self, x, target_class=None, index_in_batch=0):
        """
        x: [B, C, H, W]
        target_class: int o None (si None usa argmax del modelo)
        """
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)                    # [B, num_classes]
        if target_class is None:
            target = logits.argmax(dim=1)
        else:
            target = torch.tensor([target_class] * x.size(0), device=logits.device)

        # Tomar el logit de la clase objetivo
        scores = logits.gather(1, target.view(-1,1)).squeeze()
        scores.sum().backward()                   # gradiente hacia atrás

        # Activations & Gradients
        A = self.activations                      # [B, C, H, W]
        G = self.gradients                        # [B, C, H, W]
        assert A is not None and G is not None, "Hooks no capturaron activaciones/gradientes."

        # Pesos = promedio global de gradientes en el mapa espacial
        weights = G.mean(dim=(2,3), keepdim=True) # [B, C, 1, 1]
        cam = (weights * A).sum(dim=1, keepdim=True)  # [B, 1, H, W]
        cam = F.relu(cam)                          # descarta contribuciones negativas
        cam = cam[index_in_batch, 0].cpu().numpy() # [H, W]

        # Normalizar a [0,1]
        cam -= cam.min()
        cam /= (cam.max() + 1e-12)
        return cam, logits.detach()

    def remove(self):
        self.fwd_handle.remove()
        self.bwd_handle.remove()

@torch.no_grad()
def collect_easy_hard_examples(dataloader, model, device, n_easy=8, n_hard=8):
    """
    'Fáciles': correctas con mayor confianza.
    'Difíciles': incorrectas (o, si no hay, correctas con menor confianza).
    Devuelve listas de (img, true, pred, conf).
    """
    model.eval()
    easy, hard = [], []
    for xb, yb in dataloader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        logits = model(xb)
        probs  = F.softmax(logits, dim=1)
        conf, pred = probs.max(dim=1)

        correct = pred.eq(yb)
        # difíciles = incorrectas primero
        for i in range(xb.size(0)):
            rec = (xb[i].cpu(), int(yb[i].cpu()), int(pred[i].cpu()), float(conf[i].cpu()))
            if not correct[i]:
                hard.append(rec)
            else:
                easy.append(rec)

        if len(easy) >= n_easy and len(hard) >= n_hard:
            break

    # si faltan difíciles, toma correctas de menor confianza
    if len(hard) < n_hard:
        easy_sorted = sorted(easy, key=lambda r: r[3])  # por confianza asc
        need = n_hard - len(hard)
        hard.extend(easy_sorted[:need])

    # fáciles = correctas de mayor confianza
    easy_sorted = sorted([r for r in easy], key=lambda r: r[3], reverse=True)
    return easy_sorted[:n_easy], hard[:n_hard]

def show_gradcam_grid(model, dataloader, device, target_layer="layer5.2.conv3",
                      n_easy=8, n_hard=8, classes=None):
    cam = GradCAM(model, target_layer_name=target_layer)

    easy, hard = collect_easy_hard_examples(dataloader, model, device, n_easy=n_easy, n_hard=n_hard)

    def _plot_group(records, title):
        cols = 4
        rows = (len(records) + cols - 1) // cols
        plt.figure(figsize=(3.4*cols, 3.2*rows))
        for i, (img, y, p, conf) in enumerate(records, 1):
            # Grad-CAM
            x = img.unsqueeze(0).to(device)
            cam_map, _ = cam(x, target_class=p, index_in_batch=0)

            # imagen original (desnormalizada)
            img_den = denormalize_img(img)
            img_rgb = to_numpy_rgb(img_den)

            # Resize CAM al tamaño de imagen
            cam_resized = cv2.resize(cam_map, (img_rgb.shape[1], img_rgb.shape[0]))
            overlay = overlay_cam_on_image(img_rgb, cam_resized, alpha=0.45)

            plt.subplot(rows, cols, i)
            plt.imshow(overlay)
            title_i = f"y={classes[y] if classes else y} | ŷ={classes[p] if classes else p}\nconf={conf:.2f}"
            plt.title(title_i, color=('green' if y==p else 'red'), fontsize=9)
            plt.axis('off')
        plt.suptitle(title, y=1.02, weight='bold')
        plt.tight_layout()
        plt.show()

    _plot_group(easy,  "Grad-CAM: ejemplos FÁCILES (correctos, alta confianza)")
    _plot_group(hard,  "Grad-CAM: ejemplos DIFÍCILES (incorrectos o baja confianza)")

    cam.remove()














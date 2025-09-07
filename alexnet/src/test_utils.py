import torch
import matplotlib.pyplot as plt

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


def denormalize_rgb(img_tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """
    img_tensor: [3,H,W] tensor normalizado; devuelve [H,W,3] en rango ~[0,1]
    """
    if img_tensor.dim() == 4:  # [N,3,H,W] -> tomamos el primero
        img_tensor = img_tensor[0]
    out = img_tensor.clone().cpu()
    for c, (m, s) in enumerate(zip(mean, std)):
        out[c] = out[c] * s + m
    out = out.permute(1, 2, 0)            # [H,W,3]
    out = out.clamp(0, 1)                 # por si hay ligeras salidas del rango
    return out


def show_batch_images_rgb(images, labels, preds=None, n=16, title=None, class_names=None):
    """
    images: lista/tensor de [3,H,W]
    labels/preds: ints; si class_names se pasa, se mostrarán los nombres.
    """
    n = min(n, len(images))
    cols = 8
    rows = (n + cols - 1) // cols

    plt.figure(figsize=(2.4*cols, 2.6*rows))
    for i in range(n):
        img = denormalize_rgb(images[i])  # [H,W,3]
        plt.subplot(rows, cols, i+1)
        plt.imshow(img)

        y = labels[i]
        y_txt = class_names[y] if class_names is not None else str(y)

        if preds is None:
            title_txt = f"y={y_txt}"
            color = 'black'
        else:
            yhat = preds[i]
            yhat_txt = class_names[yhat] if class_names is not None else str(yhat)
            ok = (yhat == y)
            title_txt = f"y={y_txt} | ŷ={yhat_txt}"
            color = 'green' if ok else 'red'

        plt.title(title_txt, color=color, fontsize=9)
        plt.axis('off')

    if title:
        plt.suptitle(title, y=1.02, weight='bold')
    plt.tight_layout()
    plt.show()

def visualize_test_predictions_rgb(model, test_loader, device='cpu', n=16, only_errors=False, class_names=None):
    """
    Corre el modelo en test_loader y muestra un grid RGB con etiqueta real y predicción.
    - only_errors=True: sólo muestras mal clasificadas.
    - class_names: lista de nombres de clase (e.g., test_loader.dataset.classes en STL-10)
    """
    if isinstance(device, str):
        device = torch.device(device)

    model.eval()
    images_to_show, labels_to_show, preds_to_show = [], [], []

    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            logits = model(xb)                # [N, C]
            preds  = torch.argmax(logits, 1)  # [N]

            mask = preds.ne(yb) if only_errors else torch.ones_like(yb, dtype=torch.bool)
            sel = mask.nonzero(as_tuple=False).squeeze(1)

            for idx in sel:
                images_to_show.append(xb[idx].cpu())              # [3,H,W]
                labels_to_show.append(int(yb[idx].cpu()))
                preds_to_show.append(int(preds[idx].cpu()))
                if len(images_to_show) >= n:
                    break
            if len(images_to_show) >= n:
                break

    if len(images_to_show) == 0:
        print("No hay ejemplos que cumplan el criterio (quizá el modelo acertó ese batch).")
        return

    title = "Test samples (pred vs true)" if not only_errors else "Errores del modelo en test"
    show_batch_images_rgb(images_to_show, labels_to_show, preds_to_show, n=n, title=title, class_names=class_names)


def c(layer, num_filters=16):
    # layer: nn.Conv2d
    weights = layer.weight.data.clone().cpu()
    weights = (weights - weights.min()) / (weights.max() - weights.min())  # normalizar [0,1]

    fig, axes = plt.subplots(1, num_filters, figsize=(num_filters, 1))
    for i in range(num_filters):
        f = weights[i]
        f = f.permute(1, 2, 0)  # [H,W,C]
        axes[i].imshow(f)
        axes[i].axis('off')
    plt.show()


def visualize_feature_maps(model, layer_idx, input_image, device='cpu', num_maps=16):
    """
    layer_idx: índice en model.features de la capa que quieres inspeccionar
    input_image: tensor [1,3,H,W] ya normalizado
    """
    activation = {}
    def hook_fn(module, input, output):
        activation['maps'] = output.detach()

    handle = model.features[layer_idx].register_forward_hook(hook_fn)

    model.eval()
    with torch.no_grad():
        _ = model(input_image.to(device))

    handle.remove()

    maps = activation['maps'].cpu()
    maps = maps[0]  # batch=1 -> [C,H,W]

    num_maps = min(num_maps, maps.size(0))
    fig, axes = plt.subplots(1, num_maps, figsize=(num_maps, 1))
    for i in range(num_maps):
        fmap = maps[i]
        fmap = (fmap - fmap.min()) / (fmap.max() - fmap.min())
        axes[i].imshow(fmap, cmap='viridis')
        axes[i].axis('off')
    plt.show()


def get_embeddings(model, dataloader, device):
    model.eval()
    embeddings = []
    labels = []

    with torch.no_grad():
        for xb, yb in dataloader:
            xb = xb.to(device)

            # Pasamos por features + avgpool y flatten
            feats = model.features(xb)
            feats = model.avgpool(feats)
            feats = torch.flatten(feats, 1)  # [N, dim]

            embeddings.append(feats.cpu())
            labels.append(yb)

    embeddings = torch.cat(embeddings).numpy()
    labels = torch.cat(labels).numpy()
    return embeddings, labels

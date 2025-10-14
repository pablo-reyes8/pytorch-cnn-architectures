
import numpy as np
import torch
import torch.nn.functional as F

try:
    import pandas as pd
except Exception:
    pd = None

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def _denorm_img(x: torch.Tensor,
                mean=IMAGENET_MEAN, std=IMAGENET_STD) -> torch.Tensor:
    mean = torch.tensor(mean, device=x.device)[:, None, None]
    std  = torch.tensor(std,  device=x.device)[:, None, None]
    y = x * std + mean
    return y.clamp(0, 1)


@torch.inference_mode()
def collect_calibration_table(model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    class_names = None,
    require_shuffle_false = True):
    """
    Recolecta por-ejemplo: y_true, y_pred, p_max, p_true, margin, entropy, correctness, miscalib, idx_global.
    IMPORTANTE: asume dataloader.shuffle == False para mapear índices a dataset.
    """
    if require_shuffle_false and getattr(dataloader, "shuffle", None) is True:
        raise ValueError("Se requiere val_loader con shuffle=False para recuperar índices correctamente.")

    model.eval()
    n_seen = 0
    rows = []

    for xb, yb in dataloader:
        bs = xb.size(0)
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        logits = model(xb)
        probs  = F.softmax(logits, dim=1)

        p_max, y_hat = probs.max(dim=1)
        correct = (y_hat == yb)

        # p_true = probabilidad asignada a la clase verdadera
        p_true = probs.gather(1, yb.view(-1,1)).squeeze(1)
        top2 = torch.topk(probs, k=2, dim=1).values
        margin = top2[:,0] - top2[:,1]

        entropy = -(probs * (probs.clamp_min(1e-12)).log()).sum(dim=1)
        miscalib = torch.where(correct, 1.0 - p_max, p_max)

        for i in range(bs):
            rows.append({
                "idx": n_seen + i,
                "y_true": int(yb[i].item()),
                "y_pred": int(y_hat[i].item()),
                "p_max": float(p_max[i].item()),
                "p_true": float(p_true[i].item()),
                "margin": float(margin[i].item()),
                "entropy": float(entropy[i].item()),
                "correct": bool(correct[i].item()),
                "miscalib": float(miscalib[i].item())})
        n_seen += bs

    n_classes = int(max(r["y_true"] for r in rows)) + 1
    if class_names is None:
        class_names = [f"class_{i}" for i in range(n_classes)]

    for r in rows:
        r["true_name"] = class_names[r["y_true"]]
        r["pred_name"] = class_names[r["y_pred"]]

    out = {"rows": rows, "class_names": class_names, "num_examples": len(rows)}
    if pd is not None:
        out["df"] = pd.DataFrame(rows)
    return out


def select_worst_calibrated(table,
    topN: int = 16,incorrect_only: bool = True,min_confidence: float = 0.8):

    """
    Selecciona Top-N peor calibrados por 'miscalib'.
    - incorrect_only=True: filtra a predicciones incorrectas
    - min_confidence: exige p_max >= umbral (útil para 'muy confiados y mal')
    """
    rows = table["rows"]
    if incorrect_only:
        rows = [r for r in rows if (not r["correct"])]
    if min_confidence is not None:
        rows = [r for r in rows if r["p_max"] >= float(min_confidence)]

    rows_sorted = sorted(rows, key=lambda r: r["miscalib"], reverse=True)
    return rows_sorted[:topN]


def plot_worst_grid(table,
    worst_rows,dataset: torch.utils.data.Dataset,
    ncols: int = 4,alpha: float = 0.25,
    suptitle = "Top-N peores calibrados (incorrectos, alta confianza)",save_path = None):

    """
    Dibuja una grilla de imágenes originales con título 'true/pred p_max miscalib'.
    No usa seaborn ni estilos, una sola figura con subplots de imágenes (no gráficos).
    """
    import math
    import matplotlib.pyplot as plt
    import numpy as np

    N = len(worst_rows)
    ncols = max(1, ncols)
    nrows = int(math.ceil(N / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*3.2, nrows*3.2))
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = np.array([axes])

    for k, r in enumerate(worst_rows):
        i = r["idx"]
        img, _ = dataset[i]  
        if isinstance(img, torch.Tensor):
            vis = _denorm_img(img).cpu().numpy()
            vis = np.transpose(vis, (1,2,0))
        else:
            vis = np.asarray(img)

        ax = axes[k // ncols, k % ncols]
        ax.imshow(vis)
        ax.axis("off")
        ax.set_title(f"{r['true_name']} → {r['pred_name']}\n"
                     f"p̂={r['p_max']:.2f}  miscalib={r['miscalib']:.2f}",
                     fontsize=9)

    for k in range(N, nrows*ncols):
        axes[k // ncols, k % ncols].axis("off")

    if suptitle:
        fig.suptitle(suptitle, y=1.02, fontsize=12)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


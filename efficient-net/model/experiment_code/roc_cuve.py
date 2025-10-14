from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    roc_curve, auc,
    precision_recall_curve, average_precision_score)
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import torch

__all__ = [
    "collect_logits_labels",
    "roc_pr_ovr",
    "plot_roc_ovr",
    "plot_pr_ovr",
    "save_auc_tables"]


@torch.inference_mode()
def collect_logits_labels(model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,device: torch.device,):
    """
    Recorre el dataloader y devuelve:
      y_true: (N,) int (clase verdadera)
      y_score: (N, C) float (probabilidades)
    - Se asume que el modelo devuelve logits (antes de softmax).
    - Se aplica softmax para obtener probabilidades estables numéricamente.
    """
    model.eval()
    ys, ps = [], []
    for xb, yb in dataloader:
        xb = xb.to(device, non_blocking=True)
        logits = model(xb)
        prob = torch.softmax(logits, dim=1).cpu().numpy()
        ps.append(prob)
        ys.append(yb.cpu().numpy())
    y_true = np.concatenate(ys, axis=0)
    y_score = np.concatenate(ps, axis=0)
    return y_true, y_score


def roc_pr_ovr(y_true,
    y_score,class_names = None,):
    """
    Calcula curvas ROC y PR en esquema One-vs-Rest (OvR) por clase,
    e incluye agregados micro y macro.

    Parameters
    ----------
    y_true : (N,) ints
    y_score: (N, C) probabilidades por clase
    class_names: lista opcional de nombres (len = C)

    Returns
    -------
    dict con llaves:
      - "classes": lista de nombres de clase
      - "roc": {"per_class": {k: {"fpr","tpr","auc"}}, "micro": {...}, "macro": {...}}
      - "pr":  {"per_class": {k: {"precision","recall","ap"}}, "micro": {...}, "macro": {...}}
    """
    n_classes = y_score.shape[1]
    if class_names is None:
        class_names = [f"class_{i}" for i in range(n_classes)]

    Y = label_binarize(y_true, classes=np.arange(n_classes))  


    roc_pc, pr_pc = {}, {}
    for c in range(n_classes):
        fpr, tpr, _ = roc_curve(Y[:, c], y_score[:, c])
        roc_pc[c] = {"fpr": fpr, "tpr": tpr, "auc": auc(fpr, tpr)}

        prec, rec, _ = precision_recall_curve(Y[:, c], y_score[:, c])
        pr_pc[c] = {"precision": prec,
            "recall": rec,
            "ap": average_precision_score(Y[:, c], y_score[:, c]),}

    fpr_micro, tpr_micro, _ = roc_curve(Y.ravel(), y_score.ravel())
    roc_micro_auc = auc(fpr_micro, tpr_micro)

    prec_micro, rec_micro, _ = precision_recall_curve(Y.ravel(), y_score.ravel())
    ap_micro = average_precision_score(Y.ravel(), y_score.ravel())

    fpr_grid = np.linspace(0.0, 1.0, 1001)
    tprs = [np.interp(fpr_grid, roc_pc[c]["fpr"], roc_pc[c]["tpr"]) for c in range(n_classes)]
    tpr_macro = np.mean(np.vstack(tprs), axis=0)
    roc_macro_auc = auc(fpr_grid, tpr_macro)

    rec_grid = np.linspace(0.0, 1.0, 1001)
    precs = []
    for c in range(n_classes):
        rec = pr_pc[c]["recall"]
        prec = pr_pc[c]["precision"]
        order = np.argsort(rec)
        rec_sorted = rec[order]
        prec_sorted = prec[order]
        precs.append(np.interp(rec_grid, rec_sorted, prec_sorted))
    prec_macro = np.mean(np.vstack(precs), axis=0)
    ap_macro = float(np.mean([pr_pc[c]["ap"] for c in range(n_classes)]))

    out = {
        "classes": class_names,
        "roc": {
            "per_class": roc_pc,
            "micro": {"fpr": fpr_micro, "tpr": tpr_micro, "auc": roc_micro_auc},
            "macro": {"fpr": fpr_grid, "tpr": tpr_macro, "auc": roc_macro_auc},},
        "pr": {
            "per_class": pr_pc,
            "micro": {"precision": prec_micro, "recall": rec_micro, "ap": ap_micro},
            "macro": {"precision": prec_macro, "recall": rec_grid, "ap": ap_macro},},}
    return out


def plot_roc_ovr(res,
    per_class: bool = True,
    show_micro_macro: bool = True,
    title = "ROC OvR",out_path = None,):

    """
    Dibuja ROC OvR en una sola figura (sin subplots) y sin forzar colores/estilos.
    """
    plt.figure(figsize=(8, 6))
    classes = res["classes"]
    roc_res = res["roc"]

    if per_class:
        for c, name in enumerate(classes):
            fpr = roc_res["per_class"][c]["fpr"]
            tpr = roc_res["per_class"][c]["tpr"]
            auc_c = roc_res["per_class"][c]["auc"]
            plt.plot(fpr, tpr, label=f"{name} (AUC={auc_c:.3f})", linewidth=1.0, alpha=0.9)

    if show_micro_macro:
        plt.plot(roc_res["micro"]["fpr"], roc_res["micro"]["tpr"], linestyle="--",
                 label=f"micro (AUC={roc_res['micro']['auc']:.3f})", linewidth=2.0)
        plt.plot(roc_res["macro"]["fpr"], roc_res["macro"]["tpr"], linestyle="-.",
                 label=f"macro (AUC={roc_res['macro']['auc']:.3f})", linewidth=2.0)

    plt.plot([0, 1], [0, 1], linestyle=":", linewidth=1.0)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    if title:
        plt.title(title)
    plt.legend(loc="lower right", fontsize=8, ncol=1)
    plt.tight_layout()
    if out_path is not None:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()


def plot_pr_ovr(res,
    per_class: bool = True,show_micro_macro: bool = True,
    title = "Precision-Recall OvR",out_path = None):

    """
    Dibuja PR OvR en una sola figura (sin subplots) y sin forzar colores/estilos.
    """
    plt.figure(figsize=(8, 6))
    classes = res["classes"]
    pr_res = res["pr"]

    if per_class:
        for c, name in enumerate(classes):
            prec = pr_res["per_class"][c]["precision"]
            rec  = pr_res["per_class"][c]["recall"]
            ap_c = pr_res["per_class"][c]["ap"]
            plt.plot(rec, prec, label=f"{name} (AP={ap_c:.3f})", linewidth=1.0, alpha=0.9)

    if show_micro_macro:
        plt.plot(pr_res["micro"]["recall"], pr_res["micro"]["precision"], linestyle="--",
                 label=f"micro (AP={pr_res['micro']['ap']:.3f})", linewidth=2.0)
        plt.plot(pr_res["macro"]["recall"], pr_res["macro"]["precision"], linestyle="-.",
                 label=f"macro (mAP={pr_res['macro']['ap']:.3f})", linewidth=2.0)

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    if title:
        plt.title(title)
    plt.legend(loc="lower left", fontsize=8, ncol=1)
    plt.tight_layout()
    if out_path is not None:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()


def save_auc_tables(res,roc_csv_path = None,
    pr_csv_path = None):
    """
    Guarda tablas CSV con AUC (ROC) y AP (PR) por clase + micro/macro.
    """

    import csv
    classes = res["classes"]

    if roc_csv_path is not None:
        with open(roc_csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["class", "AUC_ROC"])
            for c, name in enumerate(classes):
                w.writerow([name, f"{res['roc']['per_class'][c]['auc']:.6f}"])
            w.writerow(["micro", f"{res['roc']['micro']['auc']:.6f}"])
            w.writerow(["macro", f"{res['roc']['macro']['auc']:.6f}"])

    if pr_csv_path is not None:
        with open(pr_csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["class", "AP_PR"])
            for c, name in enumerate(classes):
                w.writerow([name, f"{res['pr']['per_class'][c]['ap']:.6f}"])
            w.writerow(["micro", f"{res['pr']['micro']['ap']:.6f}"])
            w.writerow(["macro", f"{res['pr']['macro']['ap']:.6f}"])
from typing import Optional, List, Dict, Any, Tuple
import numpy as np
import torch
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support)

try:
    import pandas as pd
except Exception:
    pd = None


@torch.inference_mode()
def _collect_true_pred(model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,device: torch.device,):

    """Recoge (y_true, y_pred) haciendo argmax sobre los logits."""
    model.eval()
    ys, yh = [], []
    for xb, yb in dataloader:
        xb = xb.to(device, non_blocking=True)
        logits = model(xb)
        pred = logits.argmax(dim=1).cpu().numpy()
        yh.append(pred)
        ys.append(yb.cpu().numpy())
    y_true = np.concatenate(ys, axis=0)
    y_pred = np.concatenate(yh, axis=0)
    return y_true, y_pred


def classification_report_on_loader(model: torch.nn.Module,
    dataloader,device,class_names = None,digits: int = 4):
    """
    Calcula classification report completo (sklearn) sobre un DataLoader.
    Devuelve:
      - 'summary': métricas globales (acc, balanced_acc, macro F1, weighted F1)
      - 'per_class': dict por clase con precision/recall/f1/support
      - 'sk_report_dict': el output_dict bruto de sklearn (por si lo necesitas)
      - 'y_true', 'y_pred'
      - 'df' (si pandas disponible): DataFrame con el reporte por clase + filas micro/macro/weighted
    """
    y_true, y_pred = _collect_true_pred(model, dataloader, device)
    n_classes = int(np.max(y_true)) + 1
    if class_names is None:
        class_names = [f"class_{i}" for i in range(n_classes)]
    else:
        assert len(class_names) == n_classes, f"len(class_names)={len(class_names)} != n_classes={n_classes}"

    rep_dict = classification_report(y_true, y_pred, target_names=class_names, digits=digits, output_dict=True)
    acc = accuracy_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)
    macro_f1    = rep_dict.get("macro avg", {}).get("f1-score", None)
    weighted_f1 = rep_dict.get("weighted avg", {}).get("f1-score", None)

    # Per-class
    per_class = {
        cname: {
            "precision": rep_dict[cname]["precision"],
            "recall":    rep_dict[cname]["recall"],
            "f1":        rep_dict[cname]["f1-score"],
            "support":   int(rep_dict[cname]["support"])}
        for cname in class_names
        if cname in rep_dict}

    out = {
        "summary": {
            "accuracy": acc,
            "balanced_accuracy": bacc,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1},
        "per_class": per_class,
        "sk_report_dict": rep_dict,
        "y_true": y_true,
        "y_pred": y_pred}

    if pd is not None:
        rows = []
        for cname in class_names:
            if cname in rep_dict:
                r = rep_dict[cname]
                rows.append({
                    "class": cname,
                    "precision": r["precision"],
                    "recall": r["recall"],
                    "f1": r["f1-score"],
                    "support": int(r["support"])})
        for key in ["micro avg", "macro avg", "weighted avg"]:
            if key in rep_dict:
                r = rep_dict[key]
                rows.append({
                    "class": key,
                    "precision": r.get("precision", np.nan),
                    "recall": r.get("recall", np.nan),
                    "f1": r.get("f1-score", np.nan),
                    "support": int(r.get("support", 0)),
                })
        df = pd.DataFrame(rows)
        out["df"] = df

    return out

def print_report_nice(report, digits: int = 3):
    """Imprime resumen + tabla por clase (si hay pandas) o en modo texto."""
    summary = report["summary"]
    print("=== SUMMARY ===")
    print(f"Accuracy           : {summary['accuracy']:.{digits}f}")
    print(f"Balanced Accuracy  : {summary['balanced_accuracy']:.{digits}f}")
    if summary["macro_f1"] is not None:
        print(f"Macro F1           : {summary['macro_f1']:.{digits}f}")
    if summary["weighted_f1"] is not None:
        print(f"Weighted F1        : {summary['weighted_f1']:.{digits}f}")

    if "df" in report and report["df"] is not None:
        df = report["df"].copy()
        with pd.option_context("display.max_rows", None, "display.max_columns", None):
            print("\n=== PER-CLASS + AGGREGATES ===")
            print(df.to_string(index=False, float_format=lambda x: f"{x:.{digits}f}"))
    else:
        print("\n=== PER-CLASS ===")
        for cname, r in report["per_class"].items():
            print(f"{cname:20s} | P: {r['precision']:.{digits}f}  R: {r['recall']:.{digits}f}  F1: {r['f1']:.{digits}f}  (n={r['support']})")

        
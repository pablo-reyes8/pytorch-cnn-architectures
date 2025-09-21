
import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt


def evaluate_model(model, val_loader, criterion, *, device="cuda", multiclass=False, plot=True):
    model.eval()
    model.to(device)
    y_true, y_pred = [], []
    y_score_all = []
    loss_sum, n_total = 0.0, 0

    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)

            if multiclass:
                probs = torch.softmax(logits, dim=1)
                preds = probs.argmax(dim=1)
                y_score_batch = probs.detach().cpu().numpy()
                loss = criterion(logits, yb.long())
            else:
                if isinstance(criterion, torch.nn.BCEWithLogitsLoss) or (logits.ndim == 2 and logits.shape[1] == 1):
                    logit = logits.squeeze(1)
                    probs = torch.sigmoid(logit)
                    preds = (probs >= 0.5).long()
                    y_score_batch = probs.detach().cpu().numpy()
                    loss = criterion(logit, yb.float())
                else:
                    probs2 = torch.softmax(logits, dim=1)
                    preds = probs2.argmax(dim=1)
                    y_score_batch = probs2[:, 1].detach().cpu().numpy()
                    loss = criterion(logits, yb.long())

            loss_sum += loss.item() * yb.size(0)
            n_total += yb.size(0)
            y_true.extend(yb.detach().cpu().tolist())
            y_pred.extend(preds.detach().cpu().tolist())
            y_score_all.append(y_score_batch)


    avg_loss = loss_sum / n_total
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_score = np.concatenate(y_score_all, axis=0)

    report = classification_report(y_true, y_pred, digits=4)

    if multiclass:
        num_classes = y_score.shape[1]
        y_true_bin = label_binarize(y_true, classes=np.arange(num_classes))
        roc_auc = roc_auc_score(y_true_bin, y_score, average="macro", multi_class="ovr")

        if plot:
            fpr, tpr, _ = roc_curve(y_true_bin.ravel(), y_score.ravel())
            roc_auc_micro = auc(fpr, tpr)
            plt.figure()
            plt.plot(fpr, tpr, label=f"micro-avg ROC (AUC={roc_auc_micro:.3f})")
            plt.plot([0,1],[0,1],"--")
            plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
            plt.title("ROC (micro-average) — multiclass")
            plt.legend(loc="lower right"); plt.grid(True, alpha=0.2); plt.show()
    else:
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        if plot:
            plt.figure()
            plt.plot(fpr, tpr, label=f"ROC (AUC={roc_auc:.3f})")
            plt.plot([0,1],[0,1],"--")
            plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
            plt.title("ROC — binary")
            plt.legend(loc="lower right"); plt.grid(True, alpha=0.2); plt.show()


    print(f"Val Loss: {avg_loss:.4f}  |  ROC-AUC: {roc_auc:.4f}")
    print("\nClassification report\n" + report)

    return {"val_loss": avg_loss,
        "roc_auc": roc_auc,
        "report": report}


from sklearn.metrics import f1_score
import torch.nn.functional as F
import torch


def train_binary_classifier(model,
    train_loader,val_loader,
    criterion,optimizer,scheduler=None,num_epochs=10,device="cuda" , eval=True, multiclass=True):

    """
    Entrena un clasificador binario o multiclase según `multiclass`.

    - multiclass=False:
        * BCEWithLogitsLoss -> salida [B] o [B,1], preds = sigmoid>=0.5
        * CrossEntropyLoss  -> salida [B,2], preds = argmax(1)
        * F1 -> average='binary'

    - multiclass=True:
        * CrossEntropyLoss  -> salida [B,C], preds = argmax(1)
        * F1 -> average='macro'
    """

    model.to(device)
    best_val_acc = 0.0
    history = {}

    train_acc_hist, train_f1_hist = [], []
    val_acc_hist,   val_f1_hist   = [], []

    for epoch in range(num_epochs):
        model.train()
        epoch_loss, correct, total = 0.0, 0, 0
        y_true_train, y_pred_train = [], []

        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()

            logits = model(X)

            if not multiclass and isinstance(criterion, torch.nn.BCEWithLogitsLoss):
                # BINARIO con BCE
                if logits.ndim == 2 and logits.shape[1] == 1:
                    logits_ = logits.squeeze(1)  
                else:
                    logits_ = logits  
                loss = criterion(logits_, y.float())
                preds = (torch.sigmoid(logits_) >= 0.5).long()  
            else:
                # CE (binario 2-clases o multiclase C-clases)
                loss = criterion(logits, y.long())
                preds = logits.argmax(dim=1)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            batch_size = y.size(0)
            epoch_loss += loss.item() * batch_size
            correct += (preds == y.long()).sum().item()
            total += batch_size

            y_true_train.extend(y.detach().cpu().tolist())
            y_pred_train.extend(preds.detach().cpu().tolist())

        avg_train_loss = epoch_loss / total
        train_acc = correct / total
        avg_tr = "macro" if multiclass else "binary"
        train_f1 = f1_score(y_true_train, y_pred_train, average=avg_tr)

        train_acc_hist.append(train_acc)
        train_f1_hist.append(train_f1)

        if eval:
            model.eval()
            val_loss_sum, val_correct, val_total = 0.0, 0, 0
            y_true_val, y_pred_val = [], []

            with torch.no_grad():
                for Xv, yv in val_loader:
                    Xv, yv = Xv.to(device), yv.to(device)
                    logits_v = model(Xv)

                    if not multiclass and isinstance(criterion, torch.nn.BCEWithLogitsLoss):
                        if logits_v.ndim == 2 and logits_v.shape[1] == 1:
                            logits_v_ = logits_v.squeeze(1)
                        else:
                            logits_v_ = logits_v
                        loss_v = criterion(logits_v_, yv.float())
                        preds_v = (torch.sigmoid(logits_v_) >= 0.5).long()
                    else:
                        loss_v = criterion(logits_v, yv.long())
                        preds_v = logits_v.argmax(dim=1)

                    bs = yv.size(0)
                    val_loss_sum += loss_v.item() * bs
                    val_correct += (preds_v == yv.long()).sum().item()
                    val_total += bs

                    y_true_val.extend(yv.detach().cpu().tolist())
                    y_pred_val.extend(preds_v.detach().cpu().tolist())

            avg_val_loss = val_loss_sum / val_total
            val_acc = val_correct / val_total
            avg_va = "macro" if multiclass else "binary"
            val_f1 = f1_score(y_true_val, y_pred_val, average=avg_va)

            val_acc_hist.append(val_acc)
            val_f1_hist.append(val_f1)
      
        if scheduler is not None:
            scheduler.step()

        if eval and val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pt")

        if eval:
            print(f"Epoch {epoch+1}/{num_epochs} | "
                f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.3f} | Train F1({avg_tr}): {train_f1:.3f} || "
                f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.3f} | Val F1({avg_va}): {val_f1:.3f}")
            
        else:
            print(f"Epoch {epoch+1}/{num_epochs} | "
                f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.3f} | Train F1({avg_tr}): {train_f1:.3f}")

        if eval:
            model.load_state_dict(torch.load("best_model.pt"))


    history = {"train_acc": train_acc_hist,"train_f1":  train_f1_hist}
      
    if eval:
        history.update({"val_acc": val_acc_hist,"val_f1":  val_f1_hist})

    return history, model
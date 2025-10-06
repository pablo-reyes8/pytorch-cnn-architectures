from torch.nn.functional import cross_entropy
from sklearn.metrics import f1_score
from torch.amp import autocast, GradScaler
import torch

def train_inception_multiclass(
    model,
    train_loader,
    optimizer,
    scheduler=None,
    num_epochs=10,
    device="cuda",
    label_smoothing=0.0,   # (opcional)
    grad_clip=5.0,         # None para desactivar
    use_amp=False):
    """
    Entrena un clasificador multiclase.
    Si el modelo es Inception v1 y devuelve auxiliares (logits, aux1, aux2),
    usa pérdida: L = CE(main) + 0.3*(CE(aux1) + CE(aux2)) perdida del paper.

    - model: nn.Module (e.g., GoogLeNetV1(aux_logits=True))
    - train_loader: DataLoader con etiquetas enteras [0..C-1]
    - optimizer, scheduler: PyTorch
    - label_smoothing: 0.0 (paper) o >0 si deseas robustez
    - grad_clip: valor float para clip de norma; None para desactivar
    - use_amp: activa torch.cuda.amp para acelerar en GPU
    """
    device = torch.device(device)
    model.to(device)
    model.train()

    scaler = GradScaler(
        device="cuda" if device.type == "cuda" else "cpu",
        enabled=(use_amp and (device.type in {"cuda", "cpu"})))

    history = {
        "train_loss": [],
        "train_acc":  [],
        "train_f1_macro": []}

    for epoch in range(num_epochs):
        epoch_loss, correct, total = 0.0, 0, 0
        y_true_train, y_pred_train = [], []

        for X, y in train_loader:
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            use_autocast = use_amp and device.type in {"cuda", "cpu"}
            autocast_kwargs = {"device_type": device.type}

            if device.type == "cpu":
                autocast_kwargs["dtype"] = torch.bfloat16
            
            with autocast(**autocast_kwargs, enabled=use_autocast):
                out = model(X)

                if isinstance(out, tuple):
                    logits, aux1, aux2 = out
                    loss_main = cross_entropy(logits, y, label_smoothing=label_smoothing)
                    loss_aux1 = cross_entropy(aux1,  y, label_smoothing=label_smoothing)
                    loss_aux2 = cross_entropy(aux2,  y, label_smoothing=label_smoothing)
                    loss = loss_main + 0.3 * (loss_aux1 + loss_aux2) # Perdida de incpetion con aux FC layers
                    preds = logits.argmax(dim=1)

                else:
                    logits = out
                    loss = cross_entropy(logits, y, label_smoothing=label_smoothing)
                    preds = logits.argmax(dim=1)

            # Backprop (con AMP opcional)
            scaler.scale(loss).backward()
            if grad_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

            scaler.step(optimizer)
            scaler.update()

            bs = y.size(0)
            epoch_loss += loss.item() * bs
            correct += (preds == y).sum().item()
            total += bs

            y_true_train.extend(y.detach().cpu().tolist())
            y_pred_train.extend(preds.detach().cpu().tolist())

        # Step del scheduler por época (paper: decay 4% cada 8 épocas)
        if scheduler is not None:
            scheduler.step()

        avg_train_loss = epoch_loss / total
        train_acc = correct / total
        train_f1 = f1_score(y_true_train, y_pred_train, average="macro")

        history["train_loss"].append(avg_train_loss)
        history["train_acc"].append(train_acc)
        history["train_f1_macro"].append(train_f1)

        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Loss: {avg_train_loss:.4f} | Acc: {train_acc:.3f} | F1(macro): {train_f1:.3f}")

    return history, model
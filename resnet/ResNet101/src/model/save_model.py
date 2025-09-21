import torch

def save_checkpoint(model, optimizer, scheduler, epoch, best_val_acc, path="checkpoint.pth"):
    checkpoint = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler else None,
        "best_val_acc": best_val_acc,}
    torch.save(checkpoint, path)
    print(f"Checkpoint guardado en {path}")


def load_checkpoint(model, optimizer=None, scheduler=None, path="checkpoint.pth", device="cuda"):
    checkpoint = torch.load(path, map_location=device)
    
    model.load_state_dict(checkpoint["model_state"])
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    if scheduler and checkpoint["scheduler_state"] is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state"])

    epoch = checkpoint["epoch"]
    best_val_acc = checkpoint.get("best_val_acc", None)

    print(f"Checkpoint cargado desde {path}, época {epoch}")
    return epoch, best_val_acc
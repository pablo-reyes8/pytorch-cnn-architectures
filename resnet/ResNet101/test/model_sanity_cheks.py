import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

def count_params(model: nn.Module):
    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": n_total, "trainable": n_train}


@torch.no_grad()
def check_forward_shapes(model: nn.Module,*,
    num_classes, device = None,
    img_size = 224, batch_size = 32):

    """
    Crea un batch sintético y valida:
      - entrada [N,3,H,W]
      - salida [N,num_classes]
    """
    if device is None:
        device = next(model.parameters()).device
    x = torch.randn(batch_size, 3, img_size, img_size, device=device)
    y = model(x)
    assert y.dim() == 2 and y.size(1) == num_classes, \
        f"Salida esperada [N,{num_classes}], got {tuple(y.shape)}"
    
    return x.shape, y.shape

def check_gradients_one_step(model,*,
    num_classes,
    device = None,img_size = 224,batch_size = 32):

    """
    Hace un paso de entrenamiento con datos sintéticos y verifica que existan
    gradientes finitos en parámetros clave.
    """

    if device is None:
        device = next(model.parameters()).device
    model.train()
    x = torch.randn(batch_size, 3, img_size, img_size, device=device)
    target = torch.randint(0, num_classes, (batch_size,), device=device)

    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    opt.zero_grad(set_to_none=True)
    logits = model(x)
    loss = criterion(logits, target)
    loss.backward()
    opt.step()

    with torch.no_grad():
        bad = []
        for n, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                    bad.append(n)
        if bad:
            raise RuntimeError(f"Gradientes NaN/Inf en: {bad}")
        

def record_stage_shapes(model, x):
    """
    Registra shapes a la salida de layer2..layer5 para verificar downsamplings.
    """

    feats = {}
    hooks = []
    for name in ["layer2", "layer3", "layer4", "layer5"]:
        mod = getattr(model, name, None)
        if mod is None:
            continue
        hooks.append(mod.register_forward_hook(lambda m, i, o, n=name: feats.setdefault(n, o.shape)))
    try:
        _ = model(x)
    finally:
        for h in hooks:
            h.remove()
    return feats

@torch.no_grad()
def sanity_report_model(model,*,num_classes,
    device = None,img_size = 224,batch_size = 32):
    """
    Reporte de sanity del modelo: params, shapes, y downsamplings esperados.
    """

    if device is None:
        device = next(model.parameters()).device

    print("== Modelo: sanity report ==")
    p = count_params(model)
    print(f"Parámetros: total={p['total']:,} | entrenables={p['trainable']:,}")


    in_shape, out_shape = check_forward_shapes(model, num_classes=num_classes,
                                               device=device, img_size=img_size,
                                               batch_size=batch_size)
    print(f"Forward OK: in{tuple(in_shape)} → out{tuple(out_shape)}")

    x = torch.randn(batch_size, 3, img_size, img_size, device=device)
    feats = record_stage_shapes(model, x)
    for k in ["layer2", "layer3", "layer4", "layer5"]:
        if k in feats:
            print(f"{k}: {tuple(feats[k])}")
    print("[OK] sanity_report_model completado.")


def train_one_epoch(model,loader,optimizer,criterion,*,
    device,amp = True):

    """
    Entrena 1 época (silencioso) y devuelve métricas básicas.
    """

    if device is None:
        device = next(model.parameters()).device
    device_type = device.type

    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    use_amp = (amp and device_type == "cuda")
    scaler = torch.amp.GradScaler(device_type if device_type in ("cuda","cpu","mps") else "cuda", enabled=use_amp)

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            with torch.amp.autocast(device_type, enabled=True):
                logits = model(xb)
                loss = criterion(logits, yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * xb.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == yb).sum().item()
        total += yb.size(0)

    loss_epoch = running_loss / max(1, total)
    acc = 100.0 * correct / max(1, total)
    return {"loss": loss_epoch, "acc": acc}


def overfit_tiny_subset(model,
    train_loader,*,k_samples = 32,epochs = 30,
    lr = 1e-3,weight_decay = 0.0,device= None,num_classes = None):

    """
    Test clásico: ¿el modelo puede memorizar un subconjunto muy pequeño?
    Debe llegar a ~100% acc si todo está bien (transformaciones no aleatorias preferibles).
    """

    if device is None:
        device = next(model.parameters()).device

    base_ds = train_loader.dataset
    tiny_idx = list(range(min(k_samples, len(base_ds))))
    tiny_ds = Subset(base_ds, tiny_idx)
    tiny_loader = DataLoader(tiny_ds, batch_size=min(16, k_samples), shuffle=True, num_workers=0, pin_memory=True)

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    last = {}
    for ep in range(1, epochs + 1):
        train_metrics = train_one_epoch(model, tiny_loader, optimizer, criterion, device=device, amp=False)
        last = train_metrics
        if ep % max(1, epochs // 5) == 0:
            print(f"[tiny] epoch {ep:02d}/{epochs}  loss={train_metrics['loss']:.4f}  acc={train_metrics['acc']:.2f}%")

    print(f"[tiny] final: loss={last['loss']:.4f}  acc={last['acc']:.2f}%")
    return last


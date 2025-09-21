import torch
import torch.nn as nn


def test_forward_shape(model, input_shape=(1, 3, 224, 224), num_classes=37, device="cpu"):
    """
    Verifica que el forward del modelo produzca [B, num_classes].
    """

    model.eval().to(device)
    x = torch.randn(*input_shape).to(device)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (input_shape[0], num_classes), \
        f"Output shape {y.shape}, esperado {(input_shape[0], num_classes)}"
    print(f" test_forward_shape: {y.shape}")


def test_no_nan_params(model):
    """
    Revisa que no existan NaN o Inf en los parámetros del modelo.
    """
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert torch.isfinite(p).all(), f" NaN/Inf en {name}"
    print("test_no_nan_params: todos los parámetros son finitos")


def test_loss_decreases_one_step(model, criterion, optimizer, device="cpu"):
    """
    Hace un paso de optimización y chequea que la pérdida baje (aproximado).
    """
    model.train().to(device)
    x = torch.randn(4, 3, 224, 224).to(device)
    y = torch.randint(0, model.classifier[-1].out_features, (4,)).to(device)

    logits = model(x)
    loss1 = criterion(logits, y)

    optimizer.zero_grad()
    loss1.backward()
    optimizer.step()

    logits2 = model(x)
    loss2 = criterion(logits2, y)

    assert loss2.item() <= loss1.item() or abs(loss2.item() - loss1.item()) < 1e-3, \
        f" La pérdida no bajó: antes {loss1.item():.4f}, después {loss2.item():.4f}"
    print(f"test_loss_decreases_one_step: {loss1.item():.4f} -> {loss2.item():.4f}")
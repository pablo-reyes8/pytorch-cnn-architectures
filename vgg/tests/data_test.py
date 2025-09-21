import torch

def test_batch_shape(loader, expected_channels=3, img_size=224):
    """
    Verifica que las imágenes tengan [B, C, H, W] correcto.
    """
    xb, yb = next(iter(loader))
    B, C, H, W = xb.shape
    assert C == expected_channels and H == img_size and W == img_size, \
        f" Batch shape inválido: {xb.shape}"
    print(f" test_batch_shape: {xb.shape}")

def test_labels_in_range(loader, num_classes=37):
    """
    Chequea que las etiquetas estén dentro del rango correcto.
    """
    _, yb = next(iter(loader))
    assert int(yb.min()) >= 0 and int(yb.max()) < num_classes, \
        f" Labels fuera de rango: {yb.min()}..{yb.max()}"
    print(f" test_labels_in_range: {yb.min().item()}..{yb.max().item()}")

def test_no_nan_in_batch(loader):
    """
    Revisa que no existan NaN en un batch (imágenes o labels).
    """
    xb, yb = next(iter(loader))
    assert torch.isfinite(xb).all(), " NaN/Inf en imágenes"
    assert torch.isfinite(yb).all(), " NaN/Inf en labels"
    print("✅ test_no_nan_in_batch: sin NaN/Inf")
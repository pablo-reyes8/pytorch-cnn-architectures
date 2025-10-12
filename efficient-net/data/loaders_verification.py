import os, time, torch
from PIL import ImageFile
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import kornia.augmentation as KA
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any



torch.backends.cudnn.benchmark = True  
try:
    torch.set_float32_matmul_precision('high')
except Exception:
    pass

ImageFile.LOAD_TRUNCATED_IMAGES = True
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


def make_loaders(
    root: str = "data",
    img_size: Tuple[int, int] = (64, 64),
    batch_size: int = 128,
    num_workers: Optional[int] = None,
    pin_memory: Optional[bool] = None) -> Dict[str, Any]:

    """Crea DataLoaders Food-101 con tfms mínimos en CPU (normalización + resize)."""
    if num_workers is None:
        num_workers = min(4, os.cpu_count() or 2)
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    tfm = transforms.Compose([
        transforms.Resize(img_size),
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])

    train_ds = datasets.Food101(root=root, split="train", download=True, transform=tfm)
    val_ds   = datasets.Food101(root=root, split="test",  download=True, transform=tfm)

    kw = dict(num_workers=num_workers, pin_memory=pin_memory)
    if num_workers > 0:
        kw.update(dict(persistent_workers=True, prefetch_factor=2))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  **kw)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size*2, shuffle=False, **kw)

    return {"train_loader": train_loader, "val_loader": val_loader, "train_ds": train_ds, "val_ds": val_ds}


def make_gpu_aug(device: torch.device):
    """Augmentación en GPU con Kornia (evita cuellos en CPU)."""
    aug = KA.AugmentationSequential(
        KA.RandomHorizontalFlip(p=0.5),
        KA.ColorJitter(0.1, 0.1, 0.1, 0.0, p=1.0), 
        data_keys=["input"], same_on_batch=False)
    return aug.to(device)


@torch.inference_mode()
def benchmark_pipeline(
    model: torch.nn.Module,
    train_loader: DataLoader,
    device: Optional[torch.device] = None,
    gpu_aug: Optional[KA.AugmentationSequential] = None,
    iters: int = 100,
    warmup: int = 10,):

    """
    Mide tiempos medios de (a) loader, (b) aug+forward en GPU.
    - Usa sincronización CUDA para medidas reales.
    - Devuelve ms/iter y throughput (imgs/s) del tramo GPU (aug+forward).
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if gpu_aug is None:
        gpu_aug = make_gpu_aug(device)

    model.eval().to(device)
    use_cuda = device.type == "cuda"

    it_loader = iter(train_loader)

    for _ in range(warmup):
        try:
            X, y = next(it_loader)
        except StopIteration:
            it_loader = iter(train_loader)
            X, y = next(it_loader)

    t0 = time.time()
    n_batches = 0
    for _ in range(iters):
        try:
            X, y = next(it_loader)
        except StopIteration:
            it_loader = iter(train_loader)
            X, y = next(it_loader)
        n_batches += 1
    t_loader = (time.time() - t0) / max(1, n_batches)

    X = X.to(device, non_blocking=True).to(memory_format=torch.channels_last)

    if use_cuda:
        torch.cuda.synchronize()
    for _ in range(warmup):
        with torch.autocast(device_type=device.type, dtype=torch.float16 if use_cuda else torch.bfloat16):
            X2 = gpu_aug(X)
            _ = model(X2)
    if use_cuda:
        torch.cuda.synchronize()

    # Medición GPU (aug + forward) con CUDA events
    if use_cuda:
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)

    total_ms = 0.0
    n_gpu_iters = 0

    for _ in range(iters):
        if use_cuda:
            start_evt.record()
        else:
            t0 = time.time()

        with torch.autocast(device_type=device.type, dtype=torch.float16 if use_cuda else torch.bfloat16):
            X2 = gpu_aug(X)
            _ = model(X2)

        if use_cuda:
            end_evt.record()
            torch.cuda.synchronize()
            total_ms += start_evt.elapsed_time(end_evt)  # ms
        else:
            total_ms += (time.time() - t0) * 1000.0
        n_gpu_iters += 1

    ms_per_iter_gpu = total_ms / max(1, n_gpu_iters)

    bs = X.shape[0]
    imgs_per_s = (bs / (ms_per_iter_gpu / 1000.0)) if ms_per_iter_gpu > 0 else float("nan")

    return {
        "loader_ms_per_iter": t_loader * 1000.0,
        "gpu_ms_per_iter": ms_per_iter_gpu,
        "throughput_imgs_per_s": imgs_per_s,
        "batch_size": bs}
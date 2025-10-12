import os, time, torch
from typing import List, Tuple, Dict, Any, Optional
from PIL import ImageFile
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

try:
    import kornia.augmentation as KA
    HAS_KORNIA = True
except Exception:
    HAS_KORNIA = False

ImageFile.LOAD_TRUNCATED_IMAGES = True

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def make_loader(
    num_workers: int,
    prefetch_factor: int,
    root: str = "data",
    img_size: Tuple[int, int] = (200, 200),
    batch_size: int = 64,
    persistent_workers: bool = True,
    pin_memory: Optional[bool] = None,
    download: bool = True,):

    """
    Crea un DataLoader de Food-101 con transformaciones mínimas (CPU).
    """

    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    tfms = transforms.Compose([
        transforms.Resize(img_size),
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])

    ds = datasets.Food101(root=root, split="train", download=download, transform=tfms)

    kwargs = dict(
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory)

    if num_workers > 0:
        kwargs.update(dict(
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor))

    return DataLoader(ds, **kwargs)


def _next_batch_safely(it, loader):
    """Obtiene el siguiente batch; si el iterador se agota, lo reinicia."""
    try:
        return next(it), it
    except StopIteration:
        it = iter(loader)
        return next(it), it


@torch.inference_mode()
def bench_loader(
    loader: DataLoader,
    iters: int = 80,
    warmup: int = 8,):
    """
    Mide tiempo medio por iteración del DataLoader (CPU).
    Devuelve ms/iter e imágenes/seg (aprox, usando el último batch).
    """
    it = iter(loader)


    for _ in range(max(0, warmup)):
        (_, _), it = _next_batch_safely(it, loader)

    t0 = time.time()
    nb = 0
    last_bs = None
    for _ in range(max(1, iters)):
        (X, _), it = _next_batch_safely(it, loader)
        last_bs = X.shape[0]
        nb += 1
    elapsed = time.time() - t0
    s_per_iter = elapsed / nb
    ms_per_iter = 1000.0 * s_per_iter
    imgs_per_s = (last_bs / s_per_iter) if last_bs else float("nan")
    return {"loader_ms_per_iter": ms_per_iter, "loader_imgs_per_s": imgs_per_s}


def make_gpu_aug(device: torch.device):
    """
    Crea una canalización de augmentación en GPU con Kornia (opcional).
    """
    if not HAS_KORNIA:
        return None
    
    aug = KA.AugmentationSequential(
        KA.RandomHorizontalFlip(p=0.5),
        KA.ColorJitter(0.1, 0.1, 0.1, 0.0, p=1.0),  
        data_keys=["input"], same_on_batch=False)
    return aug.to(device)


@torch.inference_mode()
def bench_gpu_aug_single_batch(
    loader: DataLoader,
    device: Optional[torch.device] = None,
    warmup: int = 8,
    iters: int = 80,):

    """
    (Opcional) Mide el tiempo de AUG en GPU sobre un único batch (para aislar CPU vs. GPU).
    Requiere Kornia y CUDA. Devuelve ms/iter y throughput del tramo de augmentación únicamente.
    """

    if not HAS_KORNIA:
        return None
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        return None

    aug = make_gpu_aug(device)
    it = iter(loader)
    (X, _), it = _next_batch_safely(it, loader)
    X = X.to(device, non_blocking=True).to(memory_format=torch.channels_last)

    # Warmup
    torch.cuda.synchronize()
    for _ in range(max(0, warmup)):
        _ = aug(X)
    torch.cuda.synchronize()

    # Medición con CUDA events
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt   = torch.cuda.Event(enable_timing=True)
    total_ms = 0.0
    n = 0
    for _ in range(max(1, iters)):
        start_evt.record()
        _ = aug(X)
        end_evt.record()
        torch.cuda.synchronize()
        total_ms += start_evt.elapsed_time(end_evt)
        n += 1

    ms_per_iter = total_ms / n
    bs = X.shape[0]
    imgs_per_s = bs / (ms_per_iter / 1000.0)
    return {"gpu_aug_ms_per_iter": ms_per_iter, "gpu_aug_imgs_per_s": imgs_per_s, "batch_size": bs}


def sweep_loader_configs(
    candidates: List[Tuple[int, int]],
    batch_size: int = 64,
    img_size: Tuple[int, int] = (200, 200),
    iters: int = 80,
    warmup: int = 8,
    persistent_workers: bool = True,
    pin_memory: Optional[bool] = None,
    print_progress: bool = True,):

    """
    Ejecuta un barrido sobre (num_workers, prefetch_factor) y devuelve resultados ordenados por ms/iter.
    """

    results = []
    for nw, pf in candidates:
        try:
            loader = make_loader(
                num_workers=nw,
                prefetch_factor=pf,
                img_size=img_size,
                batch_size=batch_size,
                persistent_workers=persistent_workers,
                pin_memory=pin_memory,
            )
            stats = bench_loader(loader, iters=iters, warmup=warmup)
            row = {"num_workers": nw, "prefetch_factor": pf, **stats}
            results.append(row)
            if print_progress:
                print(f"nw={nw:>2}, pf={pf:>2} -> {stats['loader_ms_per_iter']:6.1f} ms/iter | "
                      f"{stats['loader_imgs_per_s']:7.1f} imgs/s (BS={batch_size})")
        except Exception as e:
            if print_progress:
                print(f"Config (nw={nw}, pf={pf}) falló: {e}")

    results.sort(key=lambda r: r["loader_ms_per_iter"])
    if print_progress and results:
        best = results[0]
        print("\nMejor config:",
              f"nw={best['num_workers']}, pf={best['prefetch_factor']} -> "
              f"{best['loader_ms_per_iter']:.1f} ms/iter | {best['loader_imgs_per_s']:.1f} imgs/s")
    return results
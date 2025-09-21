import torch
import os, json
from torch.utils.data import DataLoader 
from torchvision import transforms

def _compute_mean_std(loader):
    """Media y STD por canal en [0,1]."""
    mean = torch.zeros(3)
    sqmean = torch.zeros(3)
    n_pix = 0
    for xb, _ in loader:
        b, c, h, w = xb.shape
        xb = xb.view(b, c, -1)
        mean += xb.mean(dim=(0,2)) * b
        sqmean += (xb**2).mean(dim=(0,2)) * b
        n_pix += b
    mean /= n_pix
    var = sqmean / n_pix - mean**2
    std = var.clamp_min(1e-12).sqrt()
    return mean.tolist(), std.tolist()


def _compute_median_mad(loader, max_batches=None):
    """Mediana y MAD por canal (robusto). Ojo: usa muestreo; ajusta max_batches si quieres exactitud."""
    chans = [[], [], []]
    seen = 0
    for xb, _ in loader:
        xb = xb.clamp(0,1).cpu()
        b, c, h, w = xb.shape
        sample = xb.permute(0,2,3,1).reshape(-1, c)
        idx = torch.randperm(sample.size(0))[: min(10000, sample.size(0))]
        samp = sample[idx]  # [K,3]
        for j in range(3):
            chans[j].append(samp[:, j])
        seen += 1
        if max_batches and seen >= max_batches:
            break
    chans = [torch.cat(chs) for chs in chans]
    med = [torch.median(ch).item() for ch in chans]
    mad = [torch.median((ch - med[j]).abs()).item() for j, ch in enumerate(chans)]
    mad_scaled = [max(1e-6, 1.4826*m) for m in mad]
    return med, mad_scaled

def get_or_make_stats(dataset, cache_path=None, robust=False, tmp_bs=64, num_workers=2):
    """
    dataset: Dataset SIN Normalize.
    robust: False -> (mean,std), True -> (median, MAD*1.4826).
    cache_path: JSON opcional para cachear.
    """
    if cache_path:
        try:
            if os.path.exists(cache_path) and os.path.getsize(cache_path) > 0:
                with open(cache_path, "r") as f:
                    d = json.load(f)
                if d.get("robust", False) == robust and "loc" in d and "scale" in d:
                    return d["loc"], d["scale"]
        except (JSONDecodeError, OSError, KeyError, ValueError) as e:
            print(f"[WARN] Ignorando cache de stats corrupto en '{cache_path}': {e}")

    tmp_loader = DataLoader(dataset, batch_size=tmp_bs, shuffle=False, num_workers=num_workers)
    if robust:
        loc, scale = _compute_median_mad(tmp_loader, max_batches=200)
    else:
        loc, scale = _compute_mean_std(tmp_loader)

    if cache_path:
        try:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            tmp_file = cache_path + ".tmp"
            with open(tmp_file, "w") as f:
                json.dump({"robust": robust, "loc": loc, "scale": scale}, f, indent=2)
            os.replace(tmp_file, cache_path)
        except OSError as e:
            print(f"[WARN] No se pudo escribir cache de stats en '{cache_path}': {e}")

    return loc, scale


def make_transforms(loc, scale, img_size=224, aug=True):
    norm = transforms.Normalize(mean=loc, std=scale)
    if aug:
        tf = transforms.Compose([
            transforms.Resize(int(img_size*1.14)),
            transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(0.2,0.2,0.2,0.1),
            transforms.ToTensor(),
            norm])
    else:
        tf = transforms.Compose([
            transforms.Resize(int(img_size*1.14)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            norm])
    return tf
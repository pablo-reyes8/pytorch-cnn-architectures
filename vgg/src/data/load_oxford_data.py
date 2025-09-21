
from torchvision import datasets, transforms
import torch
import os, requests, pathlib , json
from urllib.parse import urlencode
from torch.utils.data import DataLoader, random_split, Subset, Dataset
from torchvision import datasets, transforms
import numpy as np


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

class OxfordPetsBinary(torch.utils.data.Dataset):
    """
    0 = cat, 1 = dog. Funciona si base_ds es OxfordIIITPet o un Subset de él.
    """
    def __init__(self, base_ds):
        if isinstance(base_ds, torch.utils.data.Subset):
            self.base = base_ds.dataset
            self.indices = list(base_ds.indices)
        else:
            self.base = base_ds
            self.indices = None

        self._species_by_base_index = None
        try:
            anns = getattr(self.base, "_anns", None)
            imgs = getattr(self.base, "images", None) or getattr(self.base, "_images", None)
            if anns is not None and imgs is not None:
                self._species_by_base_index = {}
                for i in range(len(self.base)):
                    path, _ = imgs[i]
                    fname = path.split("/")[-1].split(".")[0]
                    sp = anns.get(fname, {}).get("species", None)
                    self._species_by_base_index[i] = sp
        except Exception:
            self._species_by_base_index = None

        self._name_is_cat = None
        if self._species_by_base_index is None:
            cat_breeds = {
                "abyssinian","bengal","birman","bombay","british_shorthair",
                "egyptian_mau","maine_coon","persian","ragdoll",
                "russian_blue","siamese","sphynx" }

            classes = getattr(self.base, "classes", [str(i) for i in range(37)])
            self._name_is_cat = [1 if c.replace(" ", "_").lower() in cat_breeds else 0 for c in classes]

    def __len__(self):
        return len(self.indices) if self.indices is not None else len(self.base)

    def __getitem__(self, idx):

        base_idx = self.indices[idx] if self.indices is not None else idx
        x, y_breed = self.base[base_idx]

        if self._species_by_base_index is not None and self._species_by_base_index.get(base_idx) in (1, 2):
            y_bin = 0 if self._species_by_base_index[base_idx] == 1 else 1
        else:
            is_cat = self._name_is_cat[y_breed]
            y_bin = 0 if is_cat == 1 else 1
        return x, y_bin


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

def get_oxford_pet_loaders(
    data_dir="./data", batch_size=32, val_split=0.2, num_workers=2, seed=42,
    mode="multiclass", img_size=224, robust=False, stats_cache_path=None,
    use_cached_if_available=True):

    """
    mode: 'multiclass' (37 razas) | 'binary' (cat vs dog)
    robust=False -> Normaliza con media/STD; True -> mediana/MAD (robusta).
    stats_cache_path: ruta JSON para cachear loc/scale.
    """

    tf_tmp = transforms.Compose([
        transforms.Resize(int(img_size*1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor()])

    base_trainval_tmp = datasets.OxfordIIITPet(
        root=data_dir, split="trainval", target_types="category",
        download=True, transform=tf_tmp)

    loc, scale = get_or_make_stats(base_trainval_tmp,
        cache_path=stats_cache_path if use_cached_if_available else None,
        robust=robust, tmp_bs=64, num_workers=num_workers)

    tf_train = make_transforms(loc, scale, img_size=img_size, aug=True)
    tf_val = make_transforms(loc, scale, img_size=img_size, aug=False)


    base_trainval = datasets.OxfordIIITPet(
        root=data_dir, split="trainval", target_types="category",
        download=False, transform=tf_train)

    base_valview = datasets.OxfordIIITPet(
        root=data_dir, split="trainval", target_types="category",
        download=False, transform=tf_val)

    g = torch.Generator().manual_seed(seed)
    val_size = int(len(base_trainval) * val_split)
    train_size = len(base_trainval) - val_size
    idx_train, idx_val = random_split(range(len(base_trainval)), [train_size, val_size], generator=g)

    if mode == "binary":
        train_ds = OxfordPetsBinary(Subset(base_trainval, idx_train.indices))
        val_ds   = OxfordPetsBinary(Subset(base_valview, idx_val.indices))
        class_names, num_classes = ["cat", "dog"], 2

    elif mode == "multiclass":
        train_ds = Subset(base_trainval, idx_train.indices)
        val_ds   = Subset(base_valview, idx_val.indices)
        class_names = getattr(base_trainval, "classes", list(range(37)))
        num_classes = 37
    else:
        raise ValueError("mode debe ser 'binary' o 'multiclass'.")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, class_names, num_classes, (loc, scale)





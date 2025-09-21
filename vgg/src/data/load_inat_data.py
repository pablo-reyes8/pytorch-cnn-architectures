
from torchvision import datasets, transforms
import torch
import os, requests, pathlib , json
from urllib.parse import urlencode
from torch.utils.data import DataLoader, random_split, Subset, Dataset

def download_inat_species(taxon_name, out_dir, n=200):
    os.makedirs(out_dir, exist_ok=True)

    params = {"taxon_name": taxon_name,
        "has[photos]": "true",
        "quality_grade": "research",
        "per_page": 200,
        "page": 1,
        "order_by": "votes"}

    got = 0
    while got < n:
        url = f"https://api.inaturalist.org/v1/observations?{urlencode(params)}"
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        obs = r.json().get("results", [])

        if not obs:
          break

        for o in obs:
            photos = o.get("photos", [])
            if not photos:
              continue

            photo_url = photos[0].get("url", "").replace("square", "medium")

            if not photo_url:
              continue

            fname = f"{o['id']}_{got}.jpg"
            p = pathlib.Path(out_dir)/fname

            try:
                img = requests.get(photo_url, timeout=30)
                img.raise_for_status()
                p.write_bytes(img.content)
                got += 1
                if got >= n: break
            except Exception:
                pass

        params["page"] += 1
    return got


def get_inat2_loaders(root="./data/inat2",
    batch_size=32,val_split=0.2,
    num_workers=1,seed=42,augment=True,pin_memory=False,persistent_workers=False):

    """
    Crea DataLoaders (train, val) desde un directorio tipo ImageFolder:
        root/
          cat/
            img1.jpg ...
          dog/
            imgX.jpg ...

    - Split estratificado por clase.
    - Transforms para VGG (224x224) + normalización ImageNet.
    - Devuelve también class_names y class_weights (para CrossEntropy).

    Returns
    -------
    train_loader, val_loader, class_names, class_weights
    """

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    if augment:
        train_tfms = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.02),
            transforms.ToTensor(),
            normalize])

    else:
        train_tfms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,])

    val_tfms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize])

    base = datasets.ImageFolder(root=root)
    class_names = base.classes
    targets = base.targets

    g = torch.Generator().manual_seed(seed)
    idx_per_class = {}
    for c in range(len(class_names)):
        idx = [i for i, y in enumerate(targets) if y == c]
        idx_per_class[c] = torch.tensor(idx)[torch.randperm(len(idx), generator=g)].tolist()

    val_indices = []
    train_indices = []
    for c, idxs in idx_per_class.items():
        n_val = max(1, int(len(idxs) * val_split))
        val_indices.extend(idxs[:n_val])
        train_indices.extend(idxs[n_val:])

    train_ds = datasets.ImageFolder(root=root, transform=train_tfms)
    val_ds   = datasets.ImageFolder(root=root, transform=val_tfms)

    train_subset = Subset(train_ds, train_indices)
    val_subset   = Subset(val_ds,   val_indices)


    n_per_class = []
    for c in range(len(class_names)):
        n_c = sum(base.targets[i] == c for i in train_indices)
        n_per_class.append(n_c)

    total = sum(n_per_class)
    class_weights = torch.tensor([total / (len(class_names) * n) for n in n_per_class], dtype=torch.float)

    train_loader = DataLoader(train_subset,batch_size=batch_size,
        shuffle=True,num_workers=num_workers,pin_memory=pin_memory,
        persistent_workers=persistent_workers if num_workers > 0 else False,drop_last=False)

    val_loader = DataLoader(val_subset,
        batch_size=batch_size,shuffle=False,num_workers=num_workers,
        pin_memory=pin_memory,persistent_workers=persistent_workers if num_workers > 0 else False,drop_last=False)

    return train_loader, val_loader, class_names, class_weights
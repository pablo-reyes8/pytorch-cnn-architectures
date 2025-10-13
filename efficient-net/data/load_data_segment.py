import os, random
from collections import defaultdict
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.utils.data import Subset
from torchvision import datasets, transforms




def _invert_dict(d:dict):
    return {v: k for k, v in d.items()}

def _orig_id_to_name(orig_id: int, class_to_idx) -> str:
    # class_to_idx es nombre->id; invertimos para id->nombre
    inv = {v: k for k, v in class_to_idx.items()}
    return inv[int(orig_id)]




def make_food101_subset_loaders(
    k_classes: int = 30,
    train_frac: float = 0.8,
    batch_size: int = 64,
    img_size = (200, 200),
    seed: int = 42,
    num_workers: int = 1,
    pin_memory: bool = True,
    data_root: str = "data",
    chosen_classes= None,
    train_transforms = None,
    val_transforms = None,
    download: bool = True):
    """
    Crea DataLoaders de un subconjunto de Food-101 con K clases.
    Selecciona K clases al azar (o usa `chosen_classes`), mezcla train y test
    del dataset original y parte en (train,val) con proporción `train_frac` 
    preservando el balance por clase. Las etiquetas se remapean a [0..K-1].

    Returns
    -------
    train_loader : DataLoader
    val_loader   : DataLoader
    info         : dict con:
        - "chosen_classes": List[str]
        - "class_to_new": Dict[int,int]   (id original -> id nuevo [0..K-1])
        - "new_to_class": Dict[int,str]   (id nuevo -> nombre de clase)
        - "counts": Dict[str, Dict[int,int]]  (conteos por clase en cada split)
        - "transforms": {"train": ..., "val": ...}
        - "splits_len": {"train": int, "val": int}
    """
    assert 0.0 < train_frac < 1.0, "train_frac debe estar entre 0 y 1."

    # Semillas reproducibles (sin tocar estado global del usuario si no quiere)
    rng = random.Random(seed)
    torch.manual_seed(seed)

    # ---------------------------
    # Transforms por defecto

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD  = (0.229, 0.224, 0.225)

    if train_transforms is None:
        train_transforms = transforms.Compose([
            transforms.Resize(img_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),])

    if val_transforms is None:
        val_transforms = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    # ---------------------------
    # Cargar Food-101 base

    base_train = datasets.Food101(root=data_root, split="train", download=download, transform=train_transforms)
    base_test  = datasets.Food101(root=data_root, split="test",  download=download, transform=val_transforms)

    all_classes = list(base_train.classes) 
    class_to_idx = dict(base_train.class_to_idx)  

    # ---------------------------
    # Elegir K clases

    if chosen_classes is not None:
        if len(chosen_classes) != k_classes:
            raise ValueError(f"len(chosen_classes)={len(chosen_classes)} != k_classes={k_classes}")
        # Validación
        unknown = [c for c in chosen_classes if c not in class_to_idx]
        if unknown:
            raise ValueError(f"Clases desconocidas para Food-101: {unknown}")
        chosen_classes_sorted = sorted(chosen_classes)
    else:
        chosen_classes_sorted = sorted(rng.sample(all_classes, k_classes))

    chosen_orig_ids = {class_to_idx[c] for c in chosen_classes_sorted}

    remap = {orig: new for new, orig in enumerate(sorted(chosen_orig_ids))}


    def _get_labels(ds):
        for attr in ("_labels", "labels", "targets"):
            if hasattr(ds, attr):
                obj = getattr(ds, attr)
                # Food101 expone _labels como lista de ints
                return list(obj)
        # fallback (lento)
        return [ds[i][1] for i in range(len(ds))]

    y_train_all = _get_labels(base_train)
    y_test_all  = _get_labels(base_test)

    # ---------------------------
    # Filtrar índices por clases elegidas

    train_indices = [i for i, y in enumerate(y_train_all) if y in chosen_orig_ids]
    test_indices  = [i for i, y in enumerate(y_test_all)  if y in chosen_orig_ids]

    by_class_train = defaultdict(list)  # new_id -> [indices en base_train]
    by_class_test  = defaultdict(list)  # new_id -> [indices en base_test]

    for i in train_indices:
        by_class_train[remap[y_train_all[i]]].append(i)
    for j in test_indices:
        by_class_test[remap[y_test_all[j]]].append(j)

    # ---------------------------
    # Construir split 
    train_sel_train_base, train_sel_test_base = [], []
    val_sel_train_base,   val_sel_test_base   = [], []

    for c in range(k_classes):
        pool = []
        pool += [("train", idx) for idx in by_class_train[c]]
        pool += [("test",  idx) for idx in by_class_test[c]]
        rng.shuffle(pool)
        cut = int(len(pool) * train_frac)
        train_part, val_part = pool[:cut], pool[cut:]

        for origin, idx in train_part:
            if origin == "train":
                train_sel_train_base.append(idx)
            else:
                train_sel_test_base.append(idx)
        for origin, idx in val_part:
            if origin == "train":
                val_sel_train_base.append(idx)
            else:
                val_sel_test_base.append(idx)

    # ---------------------------
    # Subset con remapeo

    class RemappedSubset(Dataset):
        def __init__(self, base_ds, indices, remap_dict):
            self.base = base_ds
            self.indices = list(indices)
            self.remap = dict(remap_dict)
        def __len__(self):
            return len(self.indices)
        def __getitem__(self, i):
            img, y = self.base[self.indices[i]]
            return img, self.remap[int(y)]

    train_part_from_train = RemappedSubset(base_train, train_sel_train_base, remap)
    train_part_from_test  = RemappedSubset(base_test,  train_sel_test_base,  remap)
    val_part_from_train   = RemappedSubset(base_train, val_sel_train_base,   remap)
    val_part_from_test    = RemappedSubset(base_test,  val_sel_test_base,    remap)

    train_ds = ConcatDataset([train_part_from_train, train_part_from_test])
    val_ds   = ConcatDataset([val_part_from_train,  val_part_from_test])

    # ---------------------------
    # DataLoaders

    train_loader = DataLoader(train_ds, batch_size=batch_size,
        shuffle=True,num_workers=num_workers,pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,)

    val_loader = DataLoader(val_ds,batch_size=batch_size * 2,
        shuffle=False,num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),)

    # ---------------------------
    # Métricas útiles (conteos por clase)

    def fast_count(indices_list, labels_all):
        counts = defaultdict(int)
        for idx in indices_list:
            counts[remap[int(labels_all[idx])]] += 1
        return dict(counts)

    counts_train = fast_count(train_sel_train_base, y_train_all)
    counts_train.update({k: counts_train.get(k, 0) + v for k, v in fast_count(train_sel_test_base, y_test_all).items()})
    counts_val   = fast_count(val_sel_train_base, y_train_all)
    counts_val.update({k: counts_val.get(k, 0) + v for k, v in fast_count(val_sel_test_base, y_test_all).items()})

    new_to_class = {new: _orig_id_to_name(orig_id, class_to_idx) for orig_id, new in _invert_dict(remap).items()}

    info = {
        "chosen_classes": chosen_classes_sorted,
        "class_to_new": remap,                      
        "new_to_class": new_to_class,             
        "counts": {"train": counts_train, "val": counts_val},
        "transforms": {"train": train_transforms, "val": val_transforms},
        "splits_len": {"train": len(train_ds), "val": len(val_ds)}}
    
    return train_loader, val_loader, info

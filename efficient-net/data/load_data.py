import os
from typing import Tuple, Optional, Sequence, Dict, Any
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torchvision


IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD:  Tuple[float, float, float]  = (0.229, 0.224, 0.225)


def build_food101_dataloaders(
    root: str = "data",
    img_size: Tuple[int, int] = (200, 200),
    batch_size: int = 64,
    val_batch_size: Optional[int] = None,
    normalize: bool = True,
    mean: Sequence[float] = IMAGENET_MEAN,
    std: Sequence[float] = IMAGENET_STD,
    aug_hflip_p: float = 0.5,
    aug_color_jitter: Optional[Tuple[float, float, float]] = (0.1, 0.1, 0.1),  
    shuffle_train: bool = True,
    num_workers: Optional[int] = None,
    pin_memory: Optional[bool] = None,
    persistent_workers: Optional[bool] = None,
    prefetch_factor: Optional[int] = 2,
    download: bool = True) -> Dict[str, Any]:

    """
    Construye DataLoaders de Food-101 con transformaciones consistentes para train y validación.

    Args:
        root: Carpeta base donde se almacenará/leerá el dataset ("data" por defecto).
        img_size: Tamaño (H, W) al que se redimensionarán las imágenes.
        batch_size: Lote para entrenamiento.
        val_batch_size: Lote para validación (por defecto 2 * batch_size).
        normalize: Si True, aplica normalización (útil para backbones pre-entrenados en ImageNet).
        mean: Medias por canal para la normalización.
        std: Desviaciones estándar por canal para la normalización.
        aug_hflip_p: Probabilidad de `RandomHorizontalFlip` en train. Usa 0.0 para desactivar.
        aug_color_jitter: Triple (brightness, contrast, saturation). Usa None para desactivar.
        shuffle_train: Si baraja el conjunto de entrenamiento.
        num_workers: Número de workers por DataLoader. Por defecto: min(4, os.cpu_count()).
        pin_memory: Si fija memoria en host para acelerar transferencias a GPU. Por defecto True si hay CUDA.
        persistent_workers: Mantiene workers vivos entre épocas (requiere num_workers > 0).
                           Por defecto True si num_workers > 0.
        prefetch_factor: Batches prefetched por worker (solo cuando num_workers > 0).
        download: Si descarga automáticamente el dataset si no existe.

    Returns:
        dict con:
            - "train_loader": DataLoader de entrenamiento
            - "val_loader": DataLoader de validación (split="test" oficial)
            - "train_ds": Dataset de entrenamiento
            - "val_ds": Dataset de validación
            - "num_classes": Número de clases (101)
            - "classes": Lista de nombres de clases (en orden de índices)
            - "transforms": dict con tfms de "train" y "val"
    """

    if num_workers is None:
        num_workers = min(4, os.cpu_count() or 1)

    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    if persistent_workers is None:
        persistent_workers = num_workers > 0

    if val_batch_size is None:
        val_batch_size = batch_size * 2

    # --- Transforms ---
    train_tfm_list = [
        transforms.Resize(img_size)]
    
    if aug_hflip_p and aug_hflip_p > 0.0:
        train_tfm_list.append(transforms.RandomHorizontalFlip(p=aug_hflip_p))

    if aug_color_jitter is not None:
        b, c, s = aug_color_jitter
        train_tfm_list.append(transforms.ColorJitter(brightness=b, contrast=c, saturation=s))
    train_tfm_list.append(transforms.ToTensor())

    if normalize:
        train_tfm_list.append(transforms.Normalize(mean, std))
    train_tfms = transforms.Compose(train_tfm_list)

    val_tfm_list = [transforms.Resize(img_size),
        transforms.ToTensor(),]
    
    if normalize:
        val_tfm_list.append(transforms.Normalize(mean, std))
    val_tfms = transforms.Compose(val_tfm_list)

    #  Datasets 
    train_ds = datasets.Food101(root=root, split="train", download=download, transform=train_tfms)
    val_ds   = datasets.Food101(root=root, split="test",  download=download, transform=val_tfms)

    #  DataLoaders 
    loader_kwargs = dict(
        num_workers=num_workers,
        pin_memory=pin_memory)
    
    if num_workers > 0:
        loader_kwargs.update(dict(
                persistent_workers=persistent_workers,
                prefetch_factor=prefetch_factor if prefetch_factor is not None else 2))

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle_train,
        **loader_kwargs)
    
    val_loader = DataLoader(
        val_ds,
        batch_size=val_batch_size,
        shuffle=False,
        **loader_kwargs)

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "train_ds": train_ds,
        "val_ds": val_ds,
        "num_classes": len(train_ds.classes),
        "classes": train_ds.classes,
        "transforms": {"train": train_tfms, "val": val_tfms}}


def imshow_batch(images, labels, classes, n=8):
    img_grid = torchvision.utils.make_grid(images[:n], nrow=4, normalize=True)
    plt.figure(figsize=(10,5))
    plt.imshow(img_grid.permute(1,2,0))
    plt.axis('off')
    plt.title(", ".join(classes[i] for i in labels[:n]))
    plt.show()
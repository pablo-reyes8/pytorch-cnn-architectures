import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
from dataclasses import dataclass
import matplotlib.pyplot as plt
import torchvision


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)

@dataclass
class LoaderConfig:
    data_dir: str = "./data"
    batch_size: int = 128
    num_workers: int = 4
    seed: int = 42
    drop_last: bool = False  
    pin_memory: bool = False 

def set_seed_everywhere(seed: int) :
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def cifar10_transforms(train: bool) -> transforms.Compose:
    if train:
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),])


def create_cifar10_loaders(cfg: LoaderConfig):
    os.makedirs(cfg.data_dir, exist_ok=True)
    set_seed_everywhere(cfg.seed)

    train_set = datasets.CIFAR10(
        root=cfg.data_dir, train=True, download=True, transform=cifar10_transforms(train=True))
    test_set = datasets.CIFAR10(
        root=cfg.data_dir, train=False, download=True, transform=cifar10_transforms(train=False))


    persistent = cfg.num_workers > 0
    train_loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=cfg.drop_last,
        persistent_workers=persistent,)
    
    test_loader = DataLoader(
        test_set,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=False,
        persistent_workers=persistent,)

    return train_loader, test_loader, train_set.classes


def imshow_cifar(images, labels, classes, mean=CIFAR10_MEAN, std=CIFAR10_STD, max_samples=8):
    """
    Muestra un batch de imágenes (x, y) de CIFAR-10.
    
    Args:
        images (Tensor): batch de imágenes [B, C, H, W]
        labels (Tensor): etiquetas correspondientes
        classes (list[str]): lista de nombres de clases CIFAR-10
        mean (tuple): medias usadas en normalización
        std (tuple): std usadas en normalización
        max_samples (int): número máximo de imágenes a mostrar
    """
    # Selecciona hasta max_samples imágenes
    images = images[:max_samples]
    labels = labels[:max_samples]

    # Desnormaliza
    inv_mean = torch.tensor(mean).view(3,1,1)
    inv_std = torch.tensor(std).view(3,1,1)
    images = images * inv_std + inv_mean  # desnormalizar a [0,1]

    # Convierte a grid
    grid = torchvision.utils.make_grid(images, nrow=4)  # 4 columnas
    npimg = grid.numpy().transpose((1, 2, 0))

    # Plot
    plt.figure(figsize=(10, 5))
    plt.imshow(npimg)
    plt.axis("off")
    plt.title(" | ".join(classes[labels[i]] for i in range(len(labels))))
    plt.show()




import torch
from torch.utils.data import DataLoader, ConcatDataset, random_split
from torchvision import datasets, transforms

def get_stl10_dataloaders(batch_size=64, img_size=96, train_size=10_000, num_workers=2):
    """
    Carga STL10, aplica transformaciones y devuelve DataLoaders de entrenamiento y test.

    Args:
        batch_size (int): Tamaño de batch.
        img_size (int): Dimensión de la imagen (img_size x img_size).
        train_size (int): Número de imágenes a usar en el conjunto de entrenamiento.
        num_workers (int): Hilos para DataLoader.

    Returns:
        train_loader, test_loader (torch.utils.data.DataLoader, torch.utils.data.DataLoader)
    """

    # Transformaciones
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2,
                               contrast=0.2,
                               saturation=0.2,
                               hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])])

    test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])])

    # Dataset STL10
    train_dataset = datasets.STL10(root="./data", split="train", download=True, transform=train_transform)
    test_dataset = datasets.STL10(root="./data", split="test", download=True, transform=test_transform)

    # Unir datasets
    full_dataset = ConcatDataset([train_dataset, test_dataset])

    # División train/test
    total_size = len(full_dataset)
    test_size = total_size - train_size
    train_dataset_union, test_dataset_union = random_split(full_dataset, [train_size, test_size])

    # DataLoaders
    train_loader = DataLoader(train_dataset_union, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers, pin_memory=True)

    test_loader = DataLoader(test_dataset_union, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers, pin_memory=True)

    print(f"Train set: {len(train_dataset_union)} imágenes")
    print(f"Test set: {len(test_dataset_union)} imágenes")

    # Muestra un batch para validar
    images, labels = next(iter(train_loader))
    print("Batch imágenes:", images.shape)  # (batch_size, 3, img_size, img_size)
    print("Batch labels:", labels.shape)

    return train_loader, test_loader

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from datasets import load_dataset

 
class HFCarsDataset(Dataset):
    def __init__(self, hf_subset, transform=None):
        self.hf_subset = hf_subset
        self.transform = transform

    def __len__(self):
        return len(self.hf_subset)

    def __getitem__(self, idx):
        example = self.hf_subset[idx]
        img = example["image"]   # PIL.Image
        label = example["label"] # int

        if self.transform:
            img = self.transform(img)

        return img, label


def get_stanford_cars_loaders(
    num_classes_limit: int = 20,
    img_size: int = 128,
    batch_size: int = 64,
    num_workers: int = 2):
    """
    Carga StanfordCars desde HuggingFace, filtra a las primeras `num_classes_limit`
    clases y devuelve datasets + dataloaders de train y val.

    Retorna:
      train_loader, val_loader, train_dataset, val_dataset
    """

    train_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])

    val_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])

    hf_ds = load_dataset("pkuHaowei/stanford-cars")
    hf_train = hf_ds["train"]
    hf_test  = hf_ds["test"]

    def filter_by_class(hf_subset, num_classes):
        filt = hf_subset.filter(lambda x: x["label"] < num_classes)

        def reindex(example):
            example["label"] = int(example["label"])
            return example

        return filt.map(reindex)

    hf_train = filter_by_class(hf_train, num_classes_limit)
    hf_test  = filter_by_class(hf_test,  num_classes_limit)

    train_dataset = HFCarsDataset(hf_train, transform=train_transforms)
    val_dataset   = HFCarsDataset(hf_test,  transform=val_transforms)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True)

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True)

    return train_loader, val_loader, train_dataset, val_dataset
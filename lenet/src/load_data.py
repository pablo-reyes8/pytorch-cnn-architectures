
import torch
from torchvision import datasets, transforms


def create_data():

    transform = transforms.Compose([
        transforms.Resize((32, 32)),     # LeNet-5 espera 32x32
        transforms.ToTensor(),    
        transforms.Normalize((0.1307,), (0.3081,))])

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)

    return train_loader , test_loader




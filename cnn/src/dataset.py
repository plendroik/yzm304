import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_dataloaders(batch_size=64):
    """
    CIFAR-10 veri setini indirir ve DataLoader nesnelerini dondurur.
    AlexNet/VGG gibi modeller icin goruntuler 224x224 boyutuna getirilir.
    """
    transform = transforms.Compose([
        transforms.Resize((64, 64)), # Faster on CPU, still works for AlexNet/VGG
        transforms.ToTensor(),
        # Exact CIFAR-10 mean and std values
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_data = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def get_classes():
    return ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

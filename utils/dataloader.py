import torch
from torch import nn
from torch.nn import functional as F
import torchvision


def load_data(batch_size_train: int = 64, batch_size_test: int = 64):
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                                 (0.2023, 0.1994, 0.2010))])

    train_dataset = torchvision.datasets.CIFAR10('../data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10('../data', train=False, download=True, transform=transform)

    classes_labels = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size_test, shuffle=False)
    return train_loader, test_loader, classes_labels
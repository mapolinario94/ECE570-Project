import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import timeit
import math
from models.spiking_layers import LinearLIF, Conv2dLIF
from models.spiking_models import SpikingModel
import matplotlib.pyplot as plt
import os
import ssl
if not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None):
    ssl._create_default_https_context = ssl._create_unverified_context
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def test(model: nn.Module, description: str):

    model.eval()

    test_loss = 0
    correct = 0

    with torch.no_grad():
        for images, targets in test_loader:
            images, targets = images.to(DEVICE), targets.to(DEVICE)
            output = model(images)
            test_loss += F.cross_entropy(output, targets, reduction='sum').item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(targets.data.view_as(pred)).sum()

    print(f'Test result on {description}: Avg loss is {test_loss}, Accuracy: {100.*correct/len(test_loader.dataset)}%')

    return test_loss, 100.*correct/len(test_loader.dataset)


def timesteps_performance(model: nn.Module, timesteps_list: list):
    acc_list = []
    loss_list = []
    for k in timesteps_list:
        model.timesteps = k
        loss, acc = test(model, f"Timesteps={k}")
        acc_list.append(acc)
        loss_list.append(loss)
    return loss_list, acc_list


if __name__ == "__main__":
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    test_dataset = torchvision.datasets.CIFAR10(
        '../data',
        train=False,
        download=True,
        transform=transform)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    print(test_dataset)
    batch_size_test = 100

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size_test,
        shuffle=False)

    features = nn.Sequential(
        Conv2dLIF(3, 32, kernel_size=5, padding='same', device=DEVICE, leak=1.0),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),
        Conv2dLIF(32, 64, kernel_size=5, padding='same', device=DEVICE, leak=1.0),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),
        Conv2dLIF(64, 64, kernel_size=3, padding='same', device=DEVICE, leak=1.0),
        nn.ReLU(inplace=True),
    )
    classifier = nn.Sequential(
        LinearLIF(8 * 8 * 64, 256, device=DEVICE, leak=1.0),
        nn.ReLU(inplace=True),

        LinearLIF(256, 10, cumulative=True, device=DEVICE, leak=1.0)
    )

    snn_model = SpikingModel(features, classifier, timesteps=20, device=DEVICE)
    print(snn_model)
    snn_model.load_state_dict(torch.load("../data/model_snn_20t.pth", map_location=DEVICE))

    num_timesteps = [10, 20, 30, 40, 50, 100, 200, 300]
    _, acc_list = timesteps_performance(snn_model, num_timesteps)
    plt.plot(num_timesteps, acc_list)

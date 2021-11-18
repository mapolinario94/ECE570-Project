import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import numpy as np
import timeit
import math
from models.spiking_layers import LinearLIF, Conv2dLIF
from models.spiking_models import SpikingModel
import matplotlib.pyplot as plt
from utils.metrics import timesteps_performance
# import os
# import ssl
# if not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None):
#     ssl._create_default_https_context = ssl._create_unverified_context
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


if __name__ == "__main__":
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    test_dataset = torchvision.datasets.CIFAR10(
        '../data',
        train=False,
        download=True,
        transform=transform)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    print(test_dataset)
    batch_size_test = 64

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

    # _, acc_list = timesteps_performance(snn_model, num_timesteps)
    # acc_list = list(np.load("temp_max.npy"))
    acc_list = []
    for scaling_factor in [1, 0.8, 0.6, 0.4, 0.2]:
        _, acc_factor = timesteps_performance(snn_model, test_loader, num_timesteps, scaling_factor)
        acc_list.append(acc_factor)

    scales = [1, 0.8, 0.6, 0.4, 0.2]
    fig = plt.figure("Acc vs Timesteps")
    plt.plot([71.75]*len(num_timesteps), "-o")
    for k in range(5):
        plt.plot(acc_list[k], "-o")
    plt.legend(["ANN", "SNN Vth", "SNN Vth*0.8", "SNN Vth*0.6", "SNN Vth*0.4", "SNN Vth*0.2"])
    plt.xlabel("No. Timesteps")
    plt.ylabel("Accuracy")
    plt.xticks(range(8), [str(k) for k in num_timesteps])
    plt.grid(True)
    plt.show()

    image_format = 'svg'  # e.g .png, .svg, etc.
    image_name = 'acc_timesteps3.svg'
    fig.savefig(image_name, format=image_format, dpi=1200)

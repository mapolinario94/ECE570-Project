import torch
from torch import nn
from torch.nn import functional as F
import torchvision
from models.spiking_layers import LinearLIF, Conv2dLIF
from models.spiking_models import SpikingModel
from models.conversion_method import spike_norm
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                      torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

train_dataset = torchvision.datasets.CIFAR10('../data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10('../data', train=False, download=True, transform=transform)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
print(train_dataset)

#%%

batch_size_train, batch_size_test = 64, 64

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size_test, shuffle=False)


class ANNModel(nn.Module):
    def __init__(self):
        super(ANNModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=8192, out_features=1024, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=1024, out_features=1024, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=1024, out_features=10, bias=False),
        )

    def forward(self, input):
        batch_size = input.shape[0]
        x = self.features(input)
        x = x.view(batch_size, -1)
        x = self.classifier(x)
        return x

features = nn.Sequential(
        Conv2dLIF(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
        nn.ReLU(inplace=True),
        nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
        Conv2dLIF(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5, inplace=False),
        Conv2dLIF(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
        nn.ReLU(inplace=True),
        nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        )

classifier = nn.Sequential(
        LinearLIF(in_features=8192, out_features=1024, bias=False),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5, inplace=False),
        LinearLIF(in_features=1024, out_features=1024, bias=False),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5, inplace=False),
        LinearLIF(in_features=1024, out_features=10, bias=False),
        )

snn_model = SpikingModel(features, classifier, timesteps=20, device=DEVICE)

#
# pretrained_ann = "../data/pre_trained_models/ann_vgg5_cifar10.pth"
# state = torch.load(pretrained_ann, map_location="cpu")
# ann_model = ANNModel()
# # print(ann_model.features[0].weight)
# cur_dict = ann_model.state_dict()
# for key in state['state_dict'].keys():
#     if key[7:] in cur_dict:
#         print(key[7:])
#         cur_dict[key[7:]] = nn.Parameter(state['state_dict'][key].data)
#
# ann_model.load_state_dict(cur_dict)
# # print(ann_model.features[0].weight)
# ann_model.to(DEVICE)
# # ann_model.load_state_dict(pretrained_ann['state_dict'])
#
def test(classifier, epoch):

    classifier.eval() # we need to set the mode for our model

    test_loss = 0
    correct = 0

    with torch.no_grad():
        for images, targets in test_loader:
            images, targets = images.to(DEVICE), targets.to(DEVICE)
            output = classifier(images)
            test_loss += F.cross_entropy(output, targets, reduction='sum').item()
            pred = output.data.max(1, keepdim=True)[1] # we get the estimate of our result by look at the largest class value
            correct += pred.eq(targets.data.view_as(pred)).sum() # sum up the corrected samples

    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    test_counter.append(len(train_loader.dataset)*epoch)

    print(f'Test result on epoch {epoch}: Avg loss is {test_loss}, Accuracy: {100.*correct/len(test_loader.dataset)}%')
#
#
train_losses = []
train_counter = []
test_losses = []
test_counter = []
max_epoch = 20
#
# test(ann_model, 1)
#
# snn_model = spike_norm(ann_model, snn_model, train_loader, DEVICE, 20)
snn_model.load_state_dict(torch.load("../data/snn_model_vgg5_20ts.pth"))
snn_model.to(DEVICE)


threshold_scaling = 0.6
snn_model.timesteps = 250
for layer in snn_model.features:
    if isinstance(layer, nn.Conv2d):
        layer.threshold.data *= threshold_scaling
for layer in snn_model.classifier:
    if isinstance(layer, nn.Linear):
        layer.threshold.data *= threshold_scaling
test(snn_model, 1)
from torch import nn
import torch
import torch.nn.functional as F
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def timesteps_performance(model: nn.Module, dataloader, timesteps_list: list, threshold_scaling: float = 1):
    for layer in model.features:
        if isinstance(layer, nn.Conv2d):
            layer.threshold.data *= threshold_scaling
    for layer in model.classifier:
        if isinstance(layer, nn.Linear):
            layer.threshold.data *= threshold_scaling

    acc_list = []
    loss_list = []
    for k in timesteps_list:
        model.timesteps = k
        loss, acc = test(model, f"Timesteps={k}", dataloader)
        acc_list.append(acc)
        loss_list.append(loss)

    for layer in model.features:
        if isinstance(layer, nn.Conv2d):
            layer.threshold.data /= threshold_scaling
    for layer in model.classifier:
        if isinstance(layer, nn.Linear):
            layer.threshold.data /= threshold_scaling

    return loss_list, acc_list


def test(model: nn.Module, description: str, test_loader):

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
        test_loss /= len(test_loader.dataset)
        accuracy = 100.*correct/len(test_loader.dataset)
        accuracy = accuracy.item()
    print(f'Test result on {description}: Avg loss is {test_loss}, Accuracy: {accuracy}%')

    return test_loss, accuracy


def train(classifier, epoch, optimizer, train_loader):

    classifier.train() # we need to set the mode for our model

    for batch_idx, (images, targets) in enumerate(train_loader):
        images, targets = images.to(DEVICE), targets.to(DEVICE)
        optimizer.zero_grad()
        output = classifier(images)
        loss = F.cross_entropy(output, targets)
        loss.backward()
        optimizer.step()

        if batch_idx % 300 == 0: # We visulize our output every 10 batches
            print(f'Epoch {epoch}: [{batch_idx*len(images)}/{len(train_loader.dataset)}] Loss: {loss.item()}')
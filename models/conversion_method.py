import torch
from torch import nn
from torch.nn import functional as F


def spike_norm(ann_model: nn.Module, snn_model: nn.Module, data_loader, device=None, timesteps=100, scale=1.0):
    missing_keys, unexpected_keys = snn_model.load_state_dict(ann_model.state_dict(), strict=False)
    print('\n Missing keys : {}\n Unexpected Keys: {}'.format(missing_keys, unexpected_keys))

    updating_layer = 0

    snn_model.eval()
    snn_model.to(device)
    with torch.no_grad():
        for idx, layer in enumerate(snn_model.features):
            if isinstance(layer, nn.Conv2d):
                max_threshold = torch.tensor(0)
                for images, targets in data_loader:

                    snn_model._init_internal_states(images.to(device))
                    mem_features = snn_model.mem_features
                    mask_features = snn_model.mask_features
                    for t in range(timesteps):
                        spk = images.to(device)
                        for idx_conv, layer_conv in enumerate(snn_model.features):
                            if isinstance(layer_conv, nn.Conv2d):
                                if idx_conv == idx:
                                    cur_threshold = torch.max(layer_conv._conv_forward(spk, layer_conv.weight, layer_conv.bias))
                                    if cur_threshold > max_threshold:
                                        max_threshold = cur_threshold
                                    break
                                spk, mem_features[idx_conv] = layer_conv(spk, mem_features[idx_conv])

                            elif isinstance(layer_conv, nn.AvgPool2d) or isinstance(layer_conv, nn.MaxPool2d):
                                spk = layer_conv(spk)
                            elif isinstance(layer_conv, nn.Dropout):
                                spk = spk * mask_features[idx_conv]
                layer.threshold.data = max_threshold*scale
                print(layer)
                print(f"Threshold: {layer.threshold}")

        for idx, layer in enumerate(snn_model.classifier):
            if isinstance(layer, nn.Linear):
                max_threshold = torch.tensor(0)
                for images, targets in data_loader:
                    batch_size = images.shape[0]

                    snn_model._init_internal_states(images.to(device))
                    mem_classifier = snn_model.mem_classifier
                    mem_features = snn_model.mem_features
                    mask_features = snn_model.mask_features
                    mask_classifier = snn_model.mask_classifier
                    for t in range(timesteps):
                        spk = images.to(device)
                        for conv_idx in range(len(snn_model.features)):
                            if isinstance(snn_model.features[conv_idx], nn.Conv2d):
                                spk, mem_features[conv_idx] = snn_model.features[conv_idx](spk, mem_features[conv_idx])
                            if isinstance(snn_model.features[conv_idx], nn.AvgPool2d) or isinstance(snn_model.features[conv_idx], nn.MaxPool2d):
                                spk = snn_model.features[conv_idx](spk)
                            if isinstance(snn_model.features[conv_idx], nn.Dropout):
                                spk = spk * mask_features[conv_idx]

                        spk = spk.view(batch_size, -1)

                        for idx_linear, layer_linear in enumerate(snn_model.classifier):
                            if isinstance(layer_linear, nn.Linear):
                                if idx_linear == idx:
                                    cur_threshold = torch.max(F.linear(spk, layer_linear.weight, layer_linear.bias))
                                    if cur_threshold > max_threshold:
                                        max_threshold = cur_threshold
                                    break
                                spk, mem_classifier[idx_linear] = layer_linear(spk, mem_classifier[idx_linear])
                            if isinstance(layer_linear, nn.Dropout):
                                spk = spk * mask_classifier[idx_linear]
                layer.threshold.data = max_threshold*scale
                print(layer)
                print(f"Threshold: {layer.threshold}")

    return snn_model


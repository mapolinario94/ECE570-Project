import torch
from torch import nn
import math
from models.spiking_layers import LinearLIF, Conv2dLIF


class SpikingModel(nn.Module):
    def __init__(self,
                 features: nn.Sequential,
                 classifier: nn.Sequential,
                 timesteps: int = 100,
                 device=None):
        """

        :param features:
        :param classifier:
        :param timesteps:
        :param device:
        """
        super(SpikingModel, self).__init__()
        self.timesteps = timesteps
        self.features = features
        self.classifier = classifier
        self._init_weights()

    def _init_internal_states(self):
        self.mem_classifier = []
        self.mem_features = []
        for layer_idx in range(len(self.classifier)):
            self.mem_classifier += [None]
        for layer_idx in range(len(self.features)):
            self.mem_features += [None]

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0,  math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, X):
        batch_size = X.shape[0]
        self._init_internal_states()
        for t in range(self.timesteps):
            spk = X
            for layer_idx in range(len(self.features)):
                if isinstance(self.features[layer_idx], nn.Conv2d):
                    spk, self.mem_features[layer_idx] = self.features[layer_idx](spk, self.mem_features[layer_idx])
                if isinstance(self.features[layer_idx], nn.MaxPool2d):
                    spk = self.features[layer_idx](spk)
            spk = spk.view(batch_size, -1)
            for layer_idx in range(len(self.classifier)):
                if isinstance(self.classifier[layer_idx], nn.Linear):
                    spk, self.mem_classifier[layer_idx] = self.classifier[layer_idx](spk, self.mem_classifier[layer_idx])
        return self.mem_classifier[-1]


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    features = nn.Sequential(
        Conv2dLIF(3, 32, kernel_size=5, padding='same', device=device, leak=1.0),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),
        Conv2dLIF(32, 64, kernel_size=5, padding='same', device=device, leak=1.0),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),
        Conv2dLIF(64, 64, kernel_size=3, padding='same', device=device, leak=1.0),
        nn.ReLU(inplace=True),
    )
    classifier = nn.Sequential(
        LinearLIF(8 * 8 * 64, 256, device=device, leak=1.0),
        nn.ReLU(inplace=True),

        LinearLIF(256, 10, cumulative=True, device=device, leak=1.0)
    )

    snn_model = SpikingModel(features, classifier, timesteps=20, device=device)
    print(snn_model)

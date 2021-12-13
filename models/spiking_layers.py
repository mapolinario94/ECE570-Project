import torch
from torch import nn
from torch.nn import functional as F
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class LinearSpike(torch.autograd.Function):
    """
    Here we use the piecewise-linear surrogate gradient as was done
    in Bellec et al. (2018).
    """
    gamma = 0.3  # Controls the dampening of the piecewise-linear surrogate gradient

    @staticmethod
    def forward(ctx, input_):
        ctx.save_for_backward(input_)
        output = torch.zeros_like(input_).to(DEVICE)
        output[input_ > 0] = 1.0
        # output = (input_ > 0).float()
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input_, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = LinearSpike.gamma * F.threshold(1.0 - torch.abs(input_), 0, 0)
        return grad * grad_input, None


class LinearLIF(nn.Linear):
    def __init__(self,
                 in_features,
                 out_features,
                 bias=False,
                 leak=1.0,
                 threshold=1.0,
                 learnable_tl=False,
                 individual_tl=False,
                 cumulative: bool = False,
                 activation=None,
                 reset: str = "soft",
                 device=None,
                 dtype=None) -> None:
        """
        Linear Spiking Layer
        :param in_features: number of input features
        :param out_features: number of ouput features
        :param bias: bias parameter. Default bias=False
        :param leak: leak parameter. Default leak=1.0
        :param threshold: threshold parameter. Default threshold=1.0
        :param cumulative: bool. if true, the layer just accumulate all the inputs received and return the membrane
        potential value. It should be true for last layer only. Default cumulative=False
        :param activation: activation function. if None, the layers use the LinearSpike class as an activation.
        Default activation=None
        :param reset: reset mechanism of the membrane potential. It could be "soft" or "hard". Default reset="soft"
        :param device: device to operate the layer. Default device=None
        :param dtype: data type. Default dtype=None
        """
        super(LinearLIF, self).__init__(in_features=in_features,
                                        out_features=out_features,
                                        bias=bias,
                                        device=device,
                                        dtype=dtype)
        self.out_features = out_features
        if individual_tl:
            self.leak = nn.Parameter(torch.ones(out_features)*leak, requires_grad=learnable_tl)
            self.threshold = nn.Parameter(torch.ones(out_features)*threshold, requires_grad=learnable_tl)
        else:
            self.leak = nn.Parameter(torch.tensor(leak), requires_grad=learnable_tl)
            self.threshold = nn.Parameter(torch.tensor(threshold), requires_grad=learnable_tl)
        self.reset = reset
        self.mem = None
        self.spikes = None
        self.cumulative = cumulative
        self.device = device

        if activation is None:
            self.activation = LinearSpike.apply
        else:
            self.activation = activation.apply

    def _init_neuron(self, input, mem):
        batch_size = input.shape[0]
        if mem is None:
            self.mem = torch.zeros(batch_size, self.out_features).to(self.device)
        else:
            self.mem = mem
        self.spikes = torch.zeros(batch_size, self.out_features).to(self.device)
        # self.mem_hist = torch.zeros(time_steps, batch_size, self.out_features).to(self.device)

    def forward(self, input: torch.Tensor, mem=None):
        self._init_neuron(input, mem)
        input_activation = F.linear(input, self.weight, self.bias)
        if not self.cumulative:
            self.mem = self.leak * self.mem + input_activation
            mem_thr = self.mem/self.threshold - 1.0
            self.spikes = self.activation(mem_thr)
            rst = self.threshold * (mem_thr > 0).float()
            if self.reset == "hard":
                self.mem = self.mem * (1 - rst / self.threshold)
            else:
                self.mem = self.mem - rst  # self.spikes*self.threshold

        else:
            self.mem = self.mem + input_activation
        return self.spikes, self.mem

    def extra_repr(self) -> str:
        return 'in_features={}, \nout_features={}, \nbias={}, \nleak={}, \nthreshold={}'.format(
            self.in_features, self.out_features, self.bias is not None, self.leak, self.threshold
        )


class Conv2dLIF(nn.Conv2d):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size,
                 leak=1.0,
                 threshold=1.0,
                 learnable_tl=False,
                 individual_tl=False,
                 width=None,
                 height=None,
                 activation=None,
                 reset: str = "soft",
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias: bool = False,
                 padding_mode: str = 'zeros',
                 device=None,
                 dtype=None
                 ) -> None:
        super(Conv2dLIF, self).__init__(in_channels=in_channels,
                                        out_channels=out_channels,
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        padding=padding,
                                        dilation=dilation,
                                        groups=groups,
                                        bias=bias,
                                        padding_mode=padding_mode,
                                        device=device,
                                        dtype=dtype)
        if individual_tl and width and height:
            self.leak = nn.Parameter(torch.ones(width, height)*leak, requires_grad=learnable_tl)
            self.threshold = nn.Parameter(torch.ones(width, height)*threshold, requires_grad=learnable_tl)
        else:
            self.leak = nn.Parameter(torch.tensor(leak), requires_grad=learnable_tl)
            self.threshold = nn.Parameter(torch.tensor(threshold), requires_grad=learnable_tl)
        self.reset = reset
        self.mem = None
        self.spikes = None
        self.device = device
        if activation is None:
            self.activation = LinearSpike.apply
        else:
            self.activation = activation.apply

    def _neuron_init(self, input: torch.Tensor, mem):
        self.batch_size = input.shape[0]
        self.width = input.shape[2]
        self.height = input.shape[3]
        if mem is None:
            self.mem = torch.zeros(self.batch_size, self.out_channels, self.width, self.height).to(self.device)
        else:
            self.mem = mem
        self.spikes = torch.zeros(self.batch_size, self.out_channels, self.width, self.height).to(self.device)

    def forward(self, input: torch.Tensor,  mem=None) -> tuple:
        features_input = self._conv_forward(input, self.weight, self.bias)
        self._neuron_init(features_input, mem)
        self.mem = self.leak * self.mem + features_input
        mem_thr = (self.mem / self.threshold) - 1.0
        self.spikes = self.activation(mem_thr)
        rst = self.threshold * (mem_thr > 0).float()
        if self.reset == "hard":
            self.mem = self.mem*(1-rst/self.threshold)
        else:
            self.mem = self.mem - rst  # self.spikes*self.threshold

        return self.spikes, self.mem


if __name__ == "__main__":
    linear = LinearLIF(in_features=10, out_features=5)
    print(linear)

    m = Conv2dLIF(16, 33, kernel_size=(3, 3), padding='same')
    input = torch.randn(20, 16, 50, 100)
    output, _ = m(input)
    print(output.shape)


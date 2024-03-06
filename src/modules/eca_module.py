from torch import nn
from math import log2
gamma = 2
b = 1

class eca_layer(nn.Module):
    """Constructs a ECA module.

    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel):
        super(eca_layer, self).__init__()
        k_size = int(abs(log2(channel) + b) / gamma)
        k_size = k_size if k_size % 2 else k_size + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        x = x.contiguous()
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y_ = y.squeeze(-1).contiguous()
        y_ = y_.permute(0, 2, 1).contiguous()
        y_ = self.conv(y_)
        y = y_.permute(0, 2, 1).contiguous()
        y = self.sigmoid(y.unsqueeze(-1))
        return x * y

        
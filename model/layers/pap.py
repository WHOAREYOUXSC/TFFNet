import torch
from torch import nn
import torch.nn.functional as F


class PowerAvgPool(nn.Module):
    """
    Power-Average Pooling (PAP)

    Performs generalized mean pooling:
        y = (Avg(x^p))^(1/p)
    """

    def __init__(self, kernel_size, stride, p=2.0):
        super(PowerAvgPool, self).__init__()

        self.kernel_size = kernel_size
        self.stride = stride
        self.p = p

    def forward(self, x):

        x = torch.pow(x, self.p)
        x = F.avg_pool2d(x, self.kernel_size, self.stride)
        x = torch.pow(x, 1.0 / self.p)

        return x
import torch
from torch import nn


class CumulativeLN(nn.LayerNorm):
    """
    Cumulative Layer Normalization.
    Performs simple layer normalization over channel dimension.
    """

    def __init__(self, dim, *args, **kwargs):
        super().__init__(dim, *args, **kwargs)

    def forward(self, x):
        """
        Args:
            x (Tensor): input of shape (batch_size, dim, time)
        """
        x = x.transpose(1, 2)
        x = super().forward(x)
        x = x.transpose(1, 2)
        return x


class GlobalLN(nn.Module):
    """
    Global Layer Normalization.
    Performs normalization over both channel and time dimensions.
    """

    def __init__(self, dim, eps=1e-05, elementwise_affine=True):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            self.gain = nn.Parameter(torch.ones(self.dim, 1))
            self.bias = nn.Parameter(torch.zeros(self.dim, 1))

    def forward(self, x):
        """
        Args:
            x (Tensor): input of shape (batch_size, dim, time)
        """
        dims = (-2, -1)

        mean = x.mean(dim=dims, keepdim=True)
        mean_x2 = (x**2).mean(dim=dims, keepdim=True)
        var = mean_x2 - mean**2

        x_norm = (x - mean) / torch.sqrt(var + self.eps)

        if self.elementwise_affine:
            x_norm = self.gain * x_norm + self.bias

        return x_norm

import torch
import torch.nn.functional as F
from torch import nn


class AudioEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, bias):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, bias=bias)
        self.gelu = nn.GELU()

    def forward(self, x):
        """
        Args:
            x (Tensor): input tensor (B, T)
        Returns:
            x (Tensor): output tensor (B, N, T)
        """
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = self.gelu(x)
        return x


class InputLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups=1, num_channels=in_channels, eps=1e-8)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, bias=bias)

    def forward(self, x):
        """
        Args:
            x (Tensor): audio spectrograms (B, N, T)
        Returns:
            x (Tensor): audio features (B, F, T)
        """
        x = self.norm(x)
        x = self.conv(x)
        return x


class OutputLayer(nn.Module):
    def __init__(self, in_channels, out_channels, num_spks, masking=False):
        super().__init__()
        self.masking = masking
        self.num_spks = num_spks

        hidden_dim = 2 * in_channels
        self.pconv = nn.Sequential(
            nn.Linear(in_channels, 2 * hidden_dim),
            nn.GLU(),
            nn.Linear(hidden_dim, out_channels),
        )

    def forward(self, x, ref):
        """
        Args:
            x (Tensor): separated audio features (B*S, F, T)
            ref (Tensor): audio features (B, F, T)
        Returns:
            x (Tensor): separated audio spectrograms (S, B, N, T)
        """
        x = x[..., : ref.size(-1)]

        x = self.pconv(x.transpose(1, 2)).transpose(1, 2)

        B_S, N, T = x.size()
        B = B_S // self.num_spks

        if self.masking:
            ref = ref.expand(self.num_spks, B, N, T).transpose(0, 1).contiguous()
            ref = ref.view(B * self.num_spks, N, T)
            x = F.relu(x) * ref

        x = x.view(B, self.num_spks, N, T)
        x = x.transpose(0, 1)  # [S, B, N, T]
        return x


class AudioDecoder(nn.Module):
    """Audio decoder with peak normalization"""

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.conv = nn.ConvTranspose1d(*args, **kwargs)

    def peak_normalize(self, tensor):
        # https://discuss.pytorch.org/t/how-to-normalize-audio-data-in-pytorch/187709/2
        tensor = tensor - torch.mean(tensor)
        return tensor / torch.max(torch.abs(tensor))

    def forward(self, x):
        """
        Args:
            x (Tensor): input tensor (B, N, T)
        Returns:
            x (Tensor): output tensor (B, T)
        """
        x = self.conv(x)
        x = x.squeeze(dim=1)

        x_norm = self.peak_normalize(x)

        return x_norm

from typing import Optional

import torch
from torch import Tensor, nn


class AudioEncoder(nn.Module):
    """
    Audio encoder block using STFT and 2D convolution
    """

    def __init__(self, num_channels: int, n_fft: int, hop_length: Optional[int] = None):
        """
        Args:
            num_channels (int): number of audio channels.
            n_fft (int): size of Fourier transform.
            hop_length (Optional[int]): the distance between neighboring sliding window frames.
        """
        super(AudioEncoder, self).__init__()

        self.n_fft = n_fft
        self.hop_length = hop_length

        self.conv = nn.Conv2d(in_channels=2, out_channels=num_channels, kernel_size=3, padding="same")

    def forward(self, x: Tensor):
        """
        Args:
            x (Tensor): audio to encode (B, T).
        Return:
            encoded (Tensor): encoded audio (B, C, T, F)
        """
        stft = torch.stft(x, n_fft=self.n_fft, hop_length=self.hop_length, return_complex=True)  # (2, B, F, T)

        encoded = torch.stack([stft.real, stft.imag], dim=1).transpose(2, 3)  # (B, 2, T, F)

        encoded = self.conv(encoded)  # (B, C, T, F)

        return encoded

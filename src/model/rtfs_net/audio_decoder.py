from typing import Optional

import torch
from torch import Tensor, nn


class AudioDecoder(nn.Module):
    """
    Audio decoder with 2D convolution and inverse STFT
    """

    def __init__(self, num_audio_channels: int, n_fft: int, hop_length: Optional[int] = None):
        """
        Args:
            num_audio_channels (int): number of audio channels.
            n_fft (int): size of Fourier transform.
            hop_length (Optional[int]): the distance between neighboring sliding window frames.
        """
        super(AudioDecoder, self).__init__()

        self.n_fft = n_fft
        self.hop_length = hop_length

        self.conv = nn.ConvTranspose2d(in_channels=num_audio_channels, out_channels=2, kernel_size=3, padding="same")

    def forward(self, separated_audio: Tensor, input_size: torch.Size):
        """
        Args:
            separated_audio (Tensor): audio to encode (B, C, T, F).
            input_size (torch.Size): shape of input audio.
        Return:
            decoded (Tensor): encoded audio (B, L).
        """
        _, length = input_size

        decoded = self.conv(separated_audio)  # (B, 2, T, F)
        decoded = torch.complex(decoded[:, 0], decoded[:, 1]).transpose(1, 2).contiguous()  # (B, F, T)

        decoded = torch.istft(decoded, n_fft=self.n_fft, hop_length=self.hop_length, length=length)  # (B, L)

        return decoded

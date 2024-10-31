from typing import Optional

import torch
from torch import nn

from src.model.rtfs_net.audio_encoder import AudioEncoder
from src.model.rtfs_net.caf_block import CAFBlock


class RTFSNet(nn.Module):
    """
    RTFS-Net: Recurrent time-frequency modelling for efficient audio-visual speech separation
    https://arxiv.org/pdf/2309.17189
    """

    def __init__(
        self,
        num_audio_channels: int,
        num_video_channels: int,
        num_heads: int,
        n_fft: int,
        hop_length: Optional[int] = None,
    ):
        """
        Args:
            num_audio_channels (int): number of audio channels.
            num_video_channels (int): number of audio channels.
            num_heads (int): number of heads in attention.
            n_fft (int): size of Fourier transform.
            hop_length (Optional[int]): the distance between neighboring sliding window frames.
        """
        super(RTFSNet, self).__init__()

        self.audio_encoder = AudioEncoder(
            num_channels=num_audio_channels,
            n_fft=n_fft,
            hop_length=hop_length,
        )

        self.caf = CAFBlock(
            num_audio_channels=num_audio_channels,
            num_video_channels=num_video_channels,
            num_heads=num_heads,
        )

    def forward(self):
        """
        Args:
        """
        pass

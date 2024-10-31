import torch
from torch import Tensor, nn


class S3Block(nn.Module):
    """
    Spectral Source Separation
    """

    def __init__(
        self,
        num_audio_channels: int,
    ):
        """
        Args:
            num_audio_channels (int): number of audio channels.
        """
        super(S3Block, self).__init__()

        self.masking = nn.Sequential(
            nn.PReLU(),
            nn.Conv2d(
                in_channels=num_audio_channels,
                out_channels=num_audio_channels,
                kernel_size=1,
            ),
            nn.ReLU(),
        )

        self.num_audio_channels = num_audio_channels

    def forward(self, encoded_audio: Tensor, fusion: Tensor):
        """
        Args:
            encoded_audio (Tensor): encoded audio (B, C, T, F).
            fusion (Tensor): logits of fusion audio and video (B, C, T, F).
        Return:
            separated (Tensor): separated audio (B, C, T, F).
        """

        mask = self.masking(fusion)  # (B, C, T, F)
        mask_real = mask[:, : self.num_audio_channels // 2 - 1, ...]  # (B, C // 2, T, F)
        mask_imag = mask[:, self.num_audio_channels // 2 :, ...]  # (B, C // 2, T, F)

        audio_real = encoded_audio[:, : self.num_audio_channels // 2 - 1, ...]  # (B, C // 2, T, F)
        audio_imag = encoded_audio[:, self.num_audio_channels // 2 :, ...]  # (B, C // 2, T, F)

        separated_real = mask_real * audio_real - mask_imag * audio_imag  # (B, C // 2, T, F)
        separated_imag = mask_real * audio_imag + mask_imag * audio_real  # (B, C // 2, T, F)

        separated = torch.cat([separated_real, separated_imag], dim=1)  # (B, C, T, F)

        return separated

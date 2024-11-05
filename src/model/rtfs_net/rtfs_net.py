from typing import Optional

import torch
from torch import Tensor, nn

from src.model.rtfs_net.audio_decoder import AudioDecoder
from src.model.rtfs_net.audio_encoder import AudioEncoder
from src.model.rtfs_net.caf_block import CAFBlock
from src.model.rtfs_net.rtfs_block import RTFSBlock
from src.model.rtfs_net.s3_block import S3Block
from src.model.rtfs_net.video_encoder import VideoEncoder
from src.model.rtfs_net.vp_block import VPBlock


class RTFSNet(nn.Module):
    """
    RTFS-Net: Recurrent time-frequency modelling for efficient audio-visual speech separation
    https://arxiv.org/pdf/2309.17189
    """

    def __init__(
        self,
        num_audio_channels: int,
        num_video_channels: int,
        num_rtfs_blocks: int,
        hidden_dim: int,
        hidden_dim_rnn: int,
        num_upsample_depth_rtfs: int,
        num_upsample_depth_vp: int,
        num_heads_caf: int,
        n_fft: int,
        n_freq: int,
        hop_length: Optional[int] = None,
    ):
        """
        Args:
            num_audio_channels (int): number of audio channels.
            num_video_channels (int): number of audio channels.
            num_rtfs_blocks (int): number of rtfs blocks.
            hidden_dim (int): hidden dim.
            hidden_dim_rnn (int): hidden dim rnn.
            num_upsample_depth_rtfs (int): number of upsampling layers in rtfs.
            num_upsample_depth_vp (int): number of upsampling layers in video processing block.
            num_heads_caf (int): number of heads in attention caf block.
            n_fft (int): size of Fourier transform.
            n_freq (int): n_freq.
            hop_length (Optional[int]): the distance between neighboring sliding window frames.
        """
        super(RTFSNet, self).__init__()

        self.audio_encoder = AudioEncoder(
            num_audio_channels=num_audio_channels,
            n_fft=n_fft,
            hop_length=hop_length,
        )
        self.audio_decoder = AudioDecoder(
            num_audio_channels=num_audio_channels,
            n_fft=n_fft,
            hop_length=hop_length,
        )

        self.video_encoder = VideoEncoder()
        self.vp_block = VPBlock(
            num_video_channels=num_video_channels,
            hidden_dim=hidden_dim,
            compression_multiplier=num_upsample_depth_vp,
        )

        self.caf = CAFBlock(
            num_audio_channels=num_audio_channels,
            num_video_channels=num_video_channels,
            num_heads=num_heads_caf,
        )

        self.s3 = S3Block(num_audio_channels=num_audio_channels)

        self.rtfs_block = RTFSBlock(
            num_audio_channels=num_audio_channels,
            hidden_dim=hidden_dim,
            hidden_dim_rnn=hidden_dim_rnn,
            compression_multiplier=num_upsample_depth_rtfs,
            n_freq=n_freq,
        )

        self.num_rtfs_blocks = num_rtfs_blocks

    def _forward(self, audio: Tensor, video: Tensor) -> Tensor:
        """
        Args:
            audio (Tensor): mixed audio (batch_size, 1, time).
            video (Tensor): video for lip reading (batch_size, 1, time, width, height).
        Returns:
            output (dict): output predicted separated sources (batch_size, time).
        """

        v0 = self.video_encoder(video)
        v1 = self.vp_block(v0)

        a0 = self.audio_encoder(audio)
        a1 = self.rtfs_block(a0)
        ar = self.caf(a1, v1)

        for _ in range(self.num_rtfs_blocks):
            ar = self.rtfs_block(a0 + ar)

        output = self.s3(a0, ar)

        sources = self.audio_decoder(output, audio.size())

        return sources

    def forward(self, mixture, video, **batch):
        """
        Args:
            mixture (Tensor): mixed audio (batch_size, time).
            video (Tensor): video for lip reading (batch_size, c_sources, time, width, height).
        Returns:
            output (dict): output dict containing predicted separated sources (batch_size, c_sources, time).
        """
        _, c_sources, _, _, _ = video.shape

        mixture = mixture.unsqueeze(1)
        video = video.unsqueeze(1)
        sources = [self._forward(mixture, video[:, :, i]) for i in range(c_sources)]

        return {"preds": torch.stack(sources, dim=1)}

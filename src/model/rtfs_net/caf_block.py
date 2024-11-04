import torch
from torch import Tensor, nn


class CAFBlock(nn.Module):
    """
    Cross-Dimensional attention fusion block
    """

    def __init__(self, num_audio_channels: int, num_video_channels: int, num_heads: int):
        """
        Args:
            num_audio_channels (int): number of audio channels.
            num_video_channels (int): number of audio channels.
            num_heads (int): number of heads in attention.
        """
        super(CAFBlock, self).__init__()

        self.num_heads = num_heads

        self.p1 = nn.Sequential(
            nn.Conv2d(
                in_channels=num_audio_channels,
                out_channels=num_audio_channels,
                kernel_size=1,
                groups=num_audio_channels,
            ),
            nn.GroupNorm(num_groups=1, num_channels=num_audio_channels),
        )
        self.p2 = nn.Sequential(
            nn.Conv2d(
                in_channels=num_audio_channels,
                out_channels=num_audio_channels,
                kernel_size=1,
                groups=num_audio_channels,
            ),
            nn.GroupNorm(num_groups=1, num_channels=num_audio_channels),
            nn.ReLU(),
        )
        self.f1 = nn.Sequential(
            nn.Conv1d(
                in_channels=num_video_channels,
                out_channels=num_audio_channels * num_heads,
                kernel_size=1,
                groups=num_audio_channels,
            ),
            nn.GroupNorm(num_groups=1, num_channels=num_audio_channels),
        )
        self.f2 = nn.Sequential(
            nn.Conv1d(
                in_channels=num_video_channels,
                out_channels=num_audio_channels,
                kernel_size=1,
                groups=num_audio_channels,
            ),
            nn.GroupNorm(num_groups=1, num_channels=num_audio_channels),
        )

    def forward(self, audio: Tensor, video: Tensor):
        """
        Args:
            audio (Tensor): tensor contains encoded audio (B, Ca, Ta, F).
            video (Tensor): tensor contains encoded video (B, Cv, Tv).
        Return:
            fusion (Tensor): output fusion tensor(B, Ca, Ta, F).
        """
        B, Ca, Ta, _ = audio.shape
        _, _, Tv = video.shape

        a_val = self.p1(audio)  # (B, Ca, Ta, F)
        a_gate = self.p2(audio)  # (B, Ca, Ta, F)

        vh = self.f1(video)  # (B, Ca * num_heads, Tv)
        vh = vh.view(B, Ca, self.num_heads, Tv)
        vm = torch.mean(vh, dim=2, keepdim=False)  # (B, Ca, Tv)
        v_attn = nn.functional.softmax(vm, dim=-1)  # (B, Ca, Tv)
        v_attn = nn.functional.interpolate(v_attn, size=Ta, mode="nearest")  # (B, Ca, Ta)

        v_key = self.f2(video)  # (B, Ca, Tv)
        v_key = nn.functional.interpolate(v_key, size=Ta, mode="nearest")  # (B, Ca, Ta)

        fusion1 = a_val * v_attn.unsqueeze(-1)
        fusion2 = a_gate * v_key.unsqueeze(-1)

        return fusion1 + fusion2

import torch
from torch import Tensor, nn


class CAFVBlock(nn.Module):
    """
    Video fusion block based on CAF block
    """

    def __init__(self, num_audio_channels: int, num_video_channels: int, num_heads: int):
        """
        Args:
            num_audio_channels (int): number of audio channels.
            num_video_channels (int): number of audio channels.
            num_heads (int): number of heads in attention.
        """
        super(CAFVBlock, self).__init__()

        self.num_heads = num_heads

        self.p1 = nn.Sequential(
            nn.Conv2d(
                in_channels=num_audio_channels,
                out_channels=num_video_channels,
                kernel_size=1,
                groups=num_audio_channels,
            ),
            nn.GroupNorm(num_groups=1, num_channels=num_video_channels),
        )
        self.p2 = nn.Sequential(
            nn.Conv2d(
                in_channels=num_audio_channels,
                out_channels=num_video_channels,
                kernel_size=1,
                groups=num_audio_channels,
            ),
            nn.GroupNorm(num_groups=1, num_channels=num_video_channels),
            nn.ReLU(),
        )
        self.f1 = nn.Sequential(
            nn.Conv1d(
                in_channels=num_video_channels,
                out_channels=num_video_channels * num_heads,
                kernel_size=1,
                groups=num_video_channels,
            ),
            nn.GroupNorm(num_groups=1, num_channels=num_video_channels * num_heads),
        )
        self.f2 = nn.Sequential(
            nn.Conv1d(
                in_channels=num_video_channels,
                out_channels=num_video_channels,
                kernel_size=1,
                groups=num_video_channels,
            ),
            nn.GroupNorm(num_groups=1, num_channels=num_video_channels),
        )

    def forward(self, audio: Tensor, video: Tensor):
        """
        Args:
            audio (Tensor): tensor contains encoded audio (B, Ca, Ta, F).
            video (Tensor): tensor contains encoded video (B, Cv, Tv).
        Return:
            video_fusion (Tensor): output fusion tensor (B, Cv, Tv).
        """
        B, _, _, F = audio.shape
        _, Cv, Tv = video.shape

        a_val = self.p1(audio)  # (B, Cv, Ta, F)
        a_gate = self.p2(audio)  # (B, Cv, Ta, F)

        a_val = nn.functional.interpolate(a_val, size=(Tv, F), mode="nearest")  # (B, Cv, Tv, F)
        a_gate = nn.functional.interpolate(a_gate, size=(Tv, F), mode="nearest")  # (B, Cv, Tv, F)

        vh = self.f1(video)  # (B, Cv * num_heads, Tv)
        vh = vh.view(B, Cv, self.num_heads, Tv)
        vm = torch.mean(vh, dim=2, keepdim=False)  # (B, Cv, Tv)
        v_attn = nn.functional.softmax(vm, dim=-1)  # (B, Cv, Tv)

        v_key = self.f2(video)  # (B, Cv, Tv)

        fusion1 = a_val * v_attn.unsqueeze(-1)  # (B, Cv, Tv, F)
        fusion2 = a_gate * v_key.unsqueeze(-1)  # (B, Cv, Tv, F)

        fusion = torch.sum(fusion1 + fusion2, dim=-1)

        return fusion + video  # (B, Cv, Tv)

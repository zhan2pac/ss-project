import torch
from sru import SRU
from torch import Tensor, nn

from src.model.rtfs_net.attention import MHSA


class TFAR(nn.Module):
    """
    Temporal-Frequency Attention Reconstruction
    """

    def __init__(
        self,
        input_size: int,
    ):
        """
        Args:
            input_size (int): input size.
        """
        super(TFAR, self).__init__()

        self.w1 = nn.Sequential(
            nn.Conv1d(
                in_channels=input_size,
                out_channels=input_size,
                kernel_size=3,
                groups=input_size,
                padding="same",
            ),
            nn.BatchNorm1d(num_features=input_size),
            nn.Sigmoid(),
        )
        self.w2 = nn.Sequential(
            nn.Conv1d(
                in_channels=input_size,
                out_channels=input_size,
                kernel_size=3,
                groups=input_size,
                padding="same",
            ),
            nn.BatchNorm1d(num_features=input_size),
        )
        self.w3 = nn.Sequential(
            nn.Conv1d(
                in_channels=input_size,
                out_channels=input_size,
                kernel_size=3,
                groups=input_size,
                padding="same",
            ),
            nn.BatchNorm1d(num_features=input_size),
        )

    def forward(self, tensor_m: Tensor, tensor_n: Tensor) -> Tensor:
        """
        Args:
            tensor_m (Tensor): tensor upsample to.
            tensor_n (Tensor): upsampled tensor.
        Return:
            output (Tensor): output tensor.
        """
        new_size = tensor_m.shape[2:]
        w1 = nn.functional.interpolate(self.w1(tensor_n), size=new_size, mode="nearest")
        w2 = self.w2(tensor_m)
        w3 = nn.functional.interpolate(self.w3(tensor_n), size=new_size, mode="nearest")

        return w1 * w2 + w3


class VPBlock(nn.Module):
    """
    Video-pProcessing block
    """

    def __init__(
        self,
        num_video_channels: int,
        hidden_dim: int,
        compression_multiplier: int,
    ):
        """
        Args:
            num_video_channels (int): number of video channels.
            hidden_dim (int): hidden dim.
            compression_multiplier (int): multiplier of compression (q in paper).
        """
        assert compression_multiplier > 0
        super(VPBlock, self).__init__()

        self.comprasion_conv = nn.Conv1d(
            in_channels=num_video_channels,
            out_channels=hidden_dim,
            kernel_size=1,
        )
        self.comprasion_phase = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    kernel_size=3,
                    stride=1 if i == 0 else 2,
                    groups=hidden_dim,
                    padding="same" if i == 0 else 0,
                )
                for i in range(compression_multiplier)
            ]
        )

        self.attention = MHSA(
            input_size=hidden_dim,
        )

        self.forward_upsampling = nn.ModuleList([TFAR(input_size=hidden_dim) for _ in range(compression_multiplier)])
        self.backward_upsampling = nn.ModuleList(
            [TFAR(input_size=hidden_dim) for _ in range(compression_multiplier - 1)]
        )

        self.conv = nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=num_video_channels,
            kernel_size=1,
        )

        self.compression_multiplier = compression_multiplier

    def forward(self, video: Tensor) -> Tensor:
        """
        Args:
            video (Tensor): tensor contains encoded audio (B, Cv, L).
        Return:
            output (Tensor): rtfs block output tensor(B, Cv, L).
        """

        compress = self.comprasion_conv(video)  # (B, hidden_dim, L)

        compressed = []
        for module in self.comprasion_phase:
            compress = module(compress)
            compressed.append(compress)

        for i, tensor in enumerate(compressed):
            if i == len(compressed) - 1:
                continue
            compress = compress + nn.functional.adaptive_avg_pool1d(tensor, compress.size()[2:])
            # (B, hidden_dim, L / 2^q)

        attention = self.attention(compress)  # (B, hidden_dim, L / 2^q)

        upsampled = [module(compressed[i], attention) for i, module in enumerate(self.forward_upsampling)]

        upsample = upsampled[-1]
        for i, module in zip(range(self.compression_multiplier - 2, -1, -1), self.backward_upsampling):
            upsample = module(upsampled[i], upsample) + compressed[i]

        output = self.conv(upsample)

        return output + video

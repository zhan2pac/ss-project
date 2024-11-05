import torch
from sru import SRU
from torch import Tensor, nn

from src.model.rtfs_net.attention import ChannelNorm, TFAttention


class DPRNNBlock(nn.Module):
    """
    Dual-Path RNN
    """

    def __init__(self, input_size: int, hidden_dim: int):
        """
        Args:
            input_size (int): input size.
            hidden_dim (int): hidden dim.
        """
        super(DPRNNBlock, self).__init__()
        self.norm = ChannelNorm((input_size, 1))
        self.unfold = nn.Unfold(
            kernel_size=(8, 1),
            stride=(1, 1),
        )
        self.sru = SRU(
            input_size=input_size * 8,
            hidden_size=hidden_dim,
            num_layers=4,
            bidirectional=True,
        )
        self.conv = nn.ConvTranspose1d(
            in_channels=2 * hidden_dim,
            out_channels=input_size,
            kernel_size=8,
            stride=1,
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): tensor (B, C, T, F).
        Return:
            output (Tensor): tensor processed by DPRNN (B, C, T, F).
        """
        B, C, T, F = x.size()

        x = self.norm(x)  # (B, C, T, F)
        unfolded = x.transpose(1, 2).contiguous().view(B * T, C, F, 1)
        unfolded = self.unfold(unfolded)  # (B * T, C * 8, F')

        unfolded = unfolded.permute(2, 0, 1)  # (F', B * T, C * 8)
        rnn, _ = self.sru(unfolded)  # (F', B * T, 2 * hidden_dim)
        rnn = rnn.permute(1, 2, 0)  # (B * T, 2 * hidden_dim, F')
        conv = self.conv(rnn)  # (B * T, C, F)
        conv = conv.view(B, T, C, F)
        conv = conv.transpose(1, 2).contiguous()  # (B, C, T, F)

        output = conv + x  # (B, C, T, F)

        return output


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
            nn.Conv2d(
                in_channels=input_size,
                out_channels=input_size,
                kernel_size=4,
                groups=input_size,
                padding="same",
            ),
            nn.GroupNorm(num_groups=1, num_channels=input_size),
            nn.Sigmoid(),
        )
        self.w2 = nn.Sequential(
            nn.Conv2d(
                in_channels=input_size,
                out_channels=input_size,
                kernel_size=4,
                groups=input_size,
                padding="same",
            ),
            nn.GroupNorm(num_groups=1, num_channels=input_size),
        )
        self.w3 = nn.Sequential(
            nn.Conv2d(
                in_channels=input_size,
                out_channels=input_size,
                kernel_size=4,
                groups=input_size,
                padding="same",
            ),
            nn.GroupNorm(num_groups=1, num_channels=input_size),
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


class RTFSBlock(nn.Module):
    """
    RTFS block
    """

    def __init__(
        self,
        num_audio_channels: int,
        hidden_dim: int,
        hidden_dim_rnn: int,
        compression_multiplier: int,
        n_freq: int,
    ):
        """
        Args:
            num_audio_channels (int): number of audio channels.
            hidden_dim (int): hidden dim.
            hidden_dim_rnn (int): hidden dim for DPRNN.
            compression_multiplier (int): multiplier of compression (q in paper).
            n_freq (int): n_freq of Fourier transform.
        """
        assert compression_multiplier > 0
        super(RTFSBlock, self).__init__()

        self.comprasion_conv = nn.Conv2d(
            in_channels=num_audio_channels,
            out_channels=hidden_dim,
            kernel_size=1,
        )
        self.comprasion_phase = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    kernel_size=4,
                    stride=1 if i == 0 else 2,
                    groups=hidden_dim,
                    padding="same" if i == 0 else 1,
                )
                for i in range(compression_multiplier)
            ]
        )

        self.freq_dprnn = DPRNNBlock(
            input_size=hidden_dim,
            hidden_dim=hidden_dim_rnn,
        )

        self.time_dprnn = DPRNNBlock(
            input_size=hidden_dim,
            hidden_dim=hidden_dim_rnn,
        )

        self.attention = TFAttention(
            input_size=hidden_dim,
            n_freq=n_freq,
        )

        self.forward_upsampling = nn.ModuleList([TFAR(input_size=hidden_dim) for _ in range(compression_multiplier)])
        self.backward_upsampling = nn.ModuleList(
            [TFAR(input_size=hidden_dim) for _ in range(compression_multiplier - 1)]
        )

        self.conv = nn.Conv2d(
            in_channels=hidden_dim,
            out_channels=num_audio_channels,
            kernel_size=1,
        )

        self.compression_multiplier = compression_multiplier

    def forward(self, audio: Tensor) -> Tensor:
        """
        Args:
            audio (Tensor): tensor contains encoded audio (B, C, T, F).
        Return:
            output (Tensor): rtfs block output tensor(B, Ca, Ta, F).
        """

        compress = self.comprasion_conv(audio)  # (B, hidden_dim, T, F)

        compressed = []
        for module in self.comprasion_phase:
            compress = module(compress)
            compressed.append(compress)

        for i, tensor in enumerate(compressed):
            if i == len(compressed) - 1:
                continue
            compress = compress + nn.functional.adaptive_avg_pool2d(tensor, compress.size()[2:])
            # (B, hidden_dim, T / 2^q, F / 2^q)

        freq_proc = self.freq_dprnn(compress)  # (B, hidden_dim, T / 2^q, F / 2^q)
        freq_proc = freq_proc.transpose(2, 3).contiguous()
        time_proc = self.time_dprnn(freq_proc)  # (B, hidden_dim, F / 2^q, T / 2^q)
        time_proc = time_proc.transpose(2, 3).contiguous()

        attention = self.attention(time_proc)  # (B, hidden_dim, T / 2^q, F / 2^q)

        upsampled = [module(compressed[i], attention) for i, module in enumerate(self.forward_upsampling)]

        upsample = upsampled[-1]
        for i, module in zip(range(self.compression_multiplier - 2, -1, -1), self.backward_upsampling):
            upsample = module(upsampled[i], upsample) + compressed[i]

        output = self.conv(upsample)

        return output + audio

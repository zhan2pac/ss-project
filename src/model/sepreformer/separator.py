import torch
import torch.nn.functional as F
from torch import nn

from .attention import RelativePositionalEncoding
from .transformers import CSTransformer, GlobalTransformer, LocalTransformer


class Separator(nn.Module):
    def __init__(
        self,
        num_spks=2,
        num_stages=4,
        num_heads=8,
        in_channels=64,
        dropout_p=0.1,
        kernel_size=65,
        dconv_kernel_size=5,
        pe_maxlen=2000,
    ):
        super().__init__()
        self.num_stages = num_stages
        self.num_heads = num_heads
        assert in_channels % num_heads == 0, "in_channels must be divisible by num_heads"

        self.pos_encoder = RelativePositionalEncoding(in_channels // num_heads, pe_maxlen)

        self.encoders = nn.ModuleList(
            [
                SeparationEncoder(
                    in_channels,
                    num_heads,
                    dropout_p,
                    kernel_size,
                    dconv_kernel_size,
                )
                for _ in range(self.num_stages)
            ]
        )

        self.bottleneck = SeparationEncoder(
            in_channels,
            num_heads,
            dropout_p,
            kernel_size,
            dconv_kernel_size,
            downsample=False,
        )
        self.splitter = SpeakerSplitter(in_channels, num_spks)

        self.fusions = nn.ModuleList(
            [nn.Conv1d(in_channels * 2, in_channels, kernel_size=1) for _ in range(self.num_stages)]
        )
        self.decoders = nn.ModuleList(
            [
                ReconstructionDecoder(
                    num_spks,
                    in_channels,
                    num_heads,
                    dropout_p,
                    kernel_size,
                )
                for _ in range(self.num_stages)
            ]
        )

    def forward(self, x):
        """
        Args:
            x (Tensor): input tensor (B, F, T)
        Returns:
            x (Tensor): separated features (B*S, F, T)
            decoder_outputs (List[Tensor]): intermediate separated features [(B*S, F, T/2) (B*S, F, T/4), ..., (B*S, F, T/2^R)]
        """
        x = self.pad_signal(x)
        input_length = x.size(-1)

        positions = torch.arange(0, input_length // 2**self.num_stages, dtype=torch.long, device=x.device)
        positions = positions[:, None] - positions[None, :]
        pos_emb = self.pos_encoder(positions)

        encoder_outputs = []
        for encoder in self.encoders:
            x, skip = encoder(x, pos_emb)  # [B, F, T/2^i]
            encoder_outputs.append(self.splitter(skip))  # [B*S, F, T/2^i]

        x, _ = self.bottleneck(x, pos_emb)  # [B, F, T/2^R]
        x = self.splitter(x)  # [B*S, F, T/2^R]

        decoder_outputs = []

        for i, (fusion, decoder) in enumerate(zip(self.fusions, self.decoders)):
            decoder_outputs.append(x)
            skip_x = encoder_outputs[self.num_stages - (i + 1)]

            x = F.upsample(x, skip_x.size(-1))  # [B*S, F, T/2^i]
            x = torch.cat([x, skip_x], dim=1)  # [B*S, F*2, T/2^i]

            x = fusion(x)  # [B*S, F, T/2^i]
            x = decoder(x, pos_emb)

        return x, decoder_outputs

    def pad_signal(self, x):
        """
        Pads a tensor to the next power of 2 in the time dimension.

        Args:
            x (Tensor): input tensor (B, F, T).
        """
        K = 2**self.num_stages
        length = x.size(-1)
        padded_length = ((length + K - 1) // K) * K

        return F.pad(x, (0, padded_length - length), value=0)


class SeparationEncoder(nn.Module):
    def __init__(
        self,
        in_channels=64,
        num_heads=8,
        dropout_p=0.1,
        kernel_size=65,
        dconv_kernel_size=5,
        downsample=True,
    ):
        super().__init__()
        self.downsample = downsample

        B_E = 2
        self.global_blocks = nn.ModuleList([GlobalTransformer(in_channels, num_heads, dropout_p) for _ in range(B_E)])
        self.local_blocks = nn.ModuleList([LocalTransformer(in_channels, kernel_size, dropout_p) for _ in range(B_E)])

        if downsample:
            self.dconv = nn.Conv1d(
                in_channels,
                in_channels,
                dconv_kernel_size,
                stride=2,
                padding=(dconv_kernel_size - 1) // 2,
                groups=in_channels,
            )
            self.batch_norm = nn.BatchNorm1d(in_channels)
            self.gelu = nn.GELU()

    def forward(self, x, pos_emb):
        """
        Args:
            x: (B, F, T)
            pos_emb: (T/2^R, T/2^R, D) where D - head dim, R - number of stages
        Returns:
            x: (B, F, T/2) if downsample=True
            skip: (B, F, T)
        """
        for block_g, block_l in zip(self.global_blocks, self.local_blocks):
            x = block_g(x, pos_emb)
            x = block_l(x)

        skip = x
        if self.downsample:
            x = self.dconv(x)
            x = self.batch_norm(x)
            x = self.gelu(x)

        return x, skip


class ReconstructionDecoder(nn.Module):
    def __init__(
        self,
        num_spks,
        in_channels=64,
        num_heads=8,
        dropout_p=0.1,
        kernel_size=65,
    ):
        super().__init__()

        B_D = 3
        self.global_blocks = nn.ModuleList([GlobalTransformer(in_channels, num_heads, dropout_p) for _ in range(B_D)])
        self.local_blocks = nn.ModuleList([LocalTransformer(in_channels, kernel_size, dropout_p) for _ in range(B_D)])
        self.cs_blocks = nn.ModuleList([CSTransformer(in_channels, num_heads, dropout_p, num_spks) for _ in range(B_D)])

    def forward(self, x, pos_emb):
        """
        Args:
            x: (B, F, T)
            pos_emb: (T/2^R, T/2^R, D) where D - head dim, R - number of stages
        Returns:
            x: (B, F, T)
        """

        for block_g, block_l, spk_attn in zip(self.global_blocks, self.local_blocks, self.cs_blocks):
            x = block_g(x, pos_emb)
            x = block_l(x)
            x = spk_attn(x)

        return x


class SpeakerSplitter(nn.Module):
    def __init__(self, in_channels, num_spks):
        super().__init__()
        self.num_spks = num_spks
        hidden_dim = 2 * in_channels * num_spks

        self.pconv1 = nn.Sequential(
            nn.Conv1d(in_channels, 2 * hidden_dim, kernel_size=1),
            nn.GLU(dim=-2),
        )
        self.pconv2 = nn.Conv1d(hidden_dim, in_channels * num_spks, kernel_size=1)
        self.norm = nn.GroupNorm(num_groups=1, num_channels=in_channels, eps=1e-8)

    def forward(self, x):
        """
        Args:
            x: (B, F, T)
        Returns:
            x: (B*S, F, T)
        """
        B, F, T = x.size()

        x = self.pconv1(x)  # [B, 2*S*F, T]
        x = self.pconv2(x)  # [B, S*F, T]

        x = x.view(B * self.num_spks, F, T).contiguous()
        x = self.norm(x)

        return x

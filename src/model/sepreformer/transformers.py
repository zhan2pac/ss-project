import torch.nn.functional as F
from torch import nn

from .attention import LayerScale, MultiHeadAttention


class GCFN(nn.Module):
    """Gated convolutional feed-forward network (GCFN)"""

    def __init__(self, in_channels, dropout_p, scale_init=1e-5):
        super().__init__()
        hidden_dim = in_channels * 3

        self.pconv1 = nn.Linear(in_channels, hidden_dim * 2)
        self.dconv = nn.Conv1d(hidden_dim * 2, hidden_dim * 2, kernel_size=3, padding=1, groups=hidden_dim * 2)
        self.glu = nn.GLU()

        self.pconv2 = nn.Sequential(
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, in_channels),
            nn.Dropout(dropout_p),
        )
        self.layer_scale = LayerScale(3, in_channels, scale_init)

    def forward(self, x):
        """
        Args:
            x: (B, T, F)
        Returns:
            x: (B, T, F)
        """
        out = self.pconv1(x)
        out = self.dconv(out.transpose(1, 2)).transpose(1, 2)
        out = self.glu(out)
        out = self.pconv2(out)

        return x + self.layer_scale(out)


class EGA(nn.Module):
    """Efficient Global Attention"""

    def __init__(self, in_channels: int, num_heads: int, dropout_p: float, scale_init=1e-5):
        super().__init__()
        self.mhsa = MultiHeadAttention(d_model=in_channels, num_heads=num_heads, dropout_p=dropout_p)
        self.layer_scale = LayerScale(3, in_channels, scale_init)
        self.gate_branch = nn.Sequential(
            nn.LayerNorm(in_channels),
            nn.Linear(in_channels, in_channels),
            nn.Sigmoid(),
        )

    def forward(self, x, pos_emb):
        """
        Args:
            x: (B, T, F)
            pos_emb: (T/2^R, T/2^R, D) where D - head dim
        Returns:
            x: (B, T, F)
        """
        time_down = pos_emb.size(0)
        time = x.size(1)

        x = x.transpose(1, 2)
        x_down = F.adaptive_avg_pool1d(x, output_size=time_down)

        attn = self.mhsa(x_down.transpose(1, 2), pos_emb)
        attn = self.layer_scale(attn).transpose(1, 2)

        out = F.upsample(attn, size=time, mode="nearest").transpose(1, 2)
        gate = self.gate_branch(x.transpose(1, 2))

        return gate * out


class CLA(nn.Module):
    """Convolutional Local Attention"""

    def __init__(self, in_channels, kernel_size, dropout_p, scale_init=1e-5):
        super().__init__()
        self.pconv1 = nn.Sequential(
            nn.Linear(in_channels, in_channels * 2),
            nn.GLU(),
        )
        self.dconv = nn.Conv1d(
            in_channels, in_channels, kernel_size, padding=(kernel_size - 1) // 2, groups=in_channels
        )
        self.pconv2 = nn.Linear(in_channels, 2 * in_channels)
        self.batch_norm = nn.BatchNorm1d(2 * in_channels)

        self.pconv3 = nn.Sequential(
            nn.GELU(),
            nn.Linear(2 * in_channels, in_channels),
        )
        self.dropout = nn.Dropout(dropout_p)
        self.layer_scale = LayerScale(3, in_channels, scale_init)

    def forward(self, x):
        """
        Args:
            x: (B, T, F)
        Returns:
            x: (B, T, F)
        """
        out = self.pconv1(x)
        out = self.dconv(out.transpose(1, 2)).transpose(1, 2)

        out = self.pconv2(out)
        out = self.batch_norm(out.transpose(1, 2)).transpose(1, 2)

        out = self.pconv3(out)
        out = self.dropout(out)

        return x + self.layer_scale(out)


class GlobalTransformer(nn.Module):
    """Global Transformer with efficient global attention (EGA)"""

    def __init__(self, in_channels: int, num_heads: int, dropout_p: float):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(in_channels)
        self.ega = EGA(num_heads=num_heads, in_channels=in_channels, dropout_p=dropout_p)

        self.layer_norm2 = nn.LayerNorm(in_channels)
        self.gcfn = GCFN(in_channels=in_channels, dropout_p=dropout_p)

    def forward(self, x, pos_emb):
        """
        Args:
            x: (B, F, T)
            pos_emb: (T/2^R, T/2^R, D) where D - head dim
        Returns:
            x: (B, T, F)
        """
        x = x.transpose(1, 2)  # [B, F, T] -> [B, T, F]

        x_norm = self.layer_norm1(x)
        out = self.ega(x_norm, pos_emb) + x

        out_norm = self.layer_norm2(out)
        out = self.gcfn(out_norm) + out

        return out


class LocalTransformer(nn.Module):
    """Local Transformer with convolutional local attention (CLA)"""

    def __init__(self, in_channels: int, kernel_size: int, dropout_p: float):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(in_channels)
        self.cla = CLA(in_channels, kernel_size, dropout_p)

        self.layer_norm2 = nn.LayerNorm(in_channels)
        self.gcfn = GCFN(in_channels, dropout_p)

    def forward(self, x):
        """
        Args:
            x: (B, T, F)
        Returns:
            x: (B, F, T)
        """
        x_norm = self.layer_norm1(x)
        out = self.cla(x_norm) + x

        out_norm = self.layer_norm2(out)
        out = self.gcfn(out_norm) + out

        out = out.transpose(1, 2)  # [B, T, F] -> [B, F, T]

        return out


class CSTransformer(nn.Module):
    """Cross-speaker (CS) Transformer"""

    def __init__(self, in_channels: int, num_heads: int, dropout_p: float, num_spks, scale_init=1e-5):
        super().__init__()
        self.num_spks = num_spks
        self.layer_norm = nn.LayerNorm(in_channels)
        self.mhsa = MultiHeadAttention(d_model=in_channels, num_heads=num_heads, dropout_p=dropout_p)
        self.layer_scale = LayerScale(3, in_channels, scale_init)

        self.feed_forward = GCFN(in_channels=in_channels, dropout_p=dropout_p)

    def forward(self, x):
        """
        Args:
            x: (B, F, T)
        Returns:
            x: (B, F, T)
        """
        B, F, T = x.size()

        x = x.view(B // self.num_spks, self.num_spks, F, T).contiguous()
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(-1, self.num_spks, F).contiguous()

        attn = self.layer_scale(self.mhsa(self.layer_norm(x), None))
        x = x + attn

        x = x.view(B // self.num_spks, T, self.num_spks, F).contiguous()
        x = x.permute([0, 2, 3, 1]).contiguous()
        x = x.view(B, F, T).contiguous()

        x = self.feed_forward(x.transpose(1, 2)).transpose(1, 2)

        return x

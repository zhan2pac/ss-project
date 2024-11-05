from math import sqrt
from typing import Optional

import torch
from torch import Tensor, nn


class ChannelNorm(nn.Module):
    """
    Channel normalization
    """

    def __init__(self, input_dim: torch.Size):
        """
        Args:
            input_dim (torch.Size): input shape.
        """
        super(ChannelNorm, self).__init__()

        self.parameter_size = 0

        self.gamma = nn.Parameter(torch.FloatTensor(*[1, input_dim[0], 1, input_dim[1]]))
        self.beta = nn.Parameter(torch.FloatTensor(*[1, input_dim[0], 1, input_dim[1]]))
        nn.init.ones_(self.gamma)
        nn.init.zeros_(self.beta)

        self.dim = (1,) if input_dim[1] == 1 else (1, 3)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): tensor (B, C, T, F).
        Return:
            normalized (Tensor): normalized processed (B, C, T, F).
        """

        mean = x.mean(self.dim, keepdim=True)

        var = torch.sqrt(x.var(self.dim, unbiased=False, keepdim=True) + 1e-5)
        normalized = ((x - mean) / var) * self.gamma + self.beta

        return normalized


class TFAttention(nn.Module):
    """
    Attention block from paper
    https://arxiv.org/pdf/2209.03952
    """

    def __init__(
        self,
        input_size: int,
        n_freq: int,
        n_head: int = 4,
        hid_chan: int = 4,
    ):
        """
        Args:
            input_size (int): input size.
            n_freq (int): n_freq of Fourier transform.
            n_head (int): number of attention heads (default: 4).
            hid_chan (int): number of hidden channels.
        """
        super(TFAttention, self).__init__()

        self.queries = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=input_size,
                        out_channels=hid_chan,
                        kernel_size=1,
                    ),
                    nn.PReLU(),
                    ChannelNorm((hid_chan, n_freq)),
                )
                for _ in range(n_head)
            ]
        )
        self.keys = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=input_size,
                        out_channels=hid_chan,
                        kernel_size=1,
                    ),
                    nn.PReLU(),
                    ChannelNorm((hid_chan, n_freq)),
                )
                for _ in range(n_head)
            ]
        )
        self.values = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=input_size,
                        out_channels=input_size // n_head,
                        kernel_size=1,
                    ),
                    nn.PReLU(),
                    ChannelNorm((input_size // n_head, n_freq)),
                )
                for _ in range(n_head)
            ]
        )
        self.proj = nn.Sequential(
            nn.Conv2d(
                in_channels=input_size,
                out_channels=input_size,
                kernel_size=1,
            ),
            nn.PReLU(),
            ChannelNorm((input_size, n_freq)),
        )

        self.n_head = n_head

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): tensor (B, C, T, F).
        Return:
            output (Tensor): tensor processed by attention (B, C, T, F).
        """
        B, C, T, F = x.size()

        queries = torch.cat([module(x) for module in self.queries], dim=0)  # (B * n_head, hid_chan, T, F)
        keys = torch.cat([module(x) for module in self.keys], dim=0)  # (B * n_head, hid_chan, T, F)
        values = torch.cat([module(x) for module in self.values], dim=0)  # (B * n_head, C / n_head, T, F)

        queries = queries.transpose(1, 2).flatten(start_dim=2)  # (B * n_head, T, hid_chan * F)
        keys = keys.transpose(1, 2).flatten(start_dim=2)  # (B * n_head, T, hid_chan * F)
        values = values.transpose(1, 2).flatten(start_dim=2)  # (B * n_head, T, C / n_head * F)

        attention = torch.matmul(queries, keys.transpose(1, 2)) / sqrt(queries.shape[-1])  # (B * n_head, T, T)
        attention = nn.functional.softmax(attention, dim=-1)  # (B * n_head, T, T)
        output = torch.matmul(attention, values)  # (B * n_head, T, C / n_head * F)
        output = output.view(B * self.n_head, T, C // self.n_head, F)
        output = output.transpose(1, 2)  # (B * n_head, C / n_head, T, F)

        output = output.view(self.n_head, B, C // self.n_head, T, F)
        output = output.transpose(0, 1).contiguous()  # (B, n_head, C / n_head, T, F)
        output = output.view(B, C, T, F)

        output = self.proj(output)  # (B, C, T, F)
        output = output + x

        return output


class FeedForward(nn.Module):
    def __init__(self, input_size: int, expansion_factor: int = 2, p_drop: float = 0.1):
        """
        Args:
            input_size (int): input size.
            expansion_factor (int): hidden_dim = expansion_factor * input_size (default: 2).
            p_drop (float): probability of dropout (default: 0.1).
        """
        super().__init__()

        hidden_dim = input_size * expansion_factor

        self.ffn = nn.Sequential(
            nn.GroupNorm(num_groups=1, num_channels=input_size),
            nn.Conv1d(in_channels=input_size, out_channels=hidden_dim, kernel_size=1),
            nn.PReLU(),
            nn.Dropout(p=p_drop),
            nn.Conv1d(in_channels=hidden_dim, out_channels=input_size, kernel_size=1),
            nn.GroupNorm(num_groups=1, num_channels=input_size),
            nn.Dropout(p=p_drop),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): Tensor of shape (B, T, C)
        Returns:
            x (Tensor): Tensor of shape (B, T, C)
        """
        return self.ffn(x) + x


class PositionalEncoding(nn.Module):
    def __init__(self, input_size: int, max_len: int = 5000):
        """
        Args:
            input_size (int): input size.
            max_len (int): max length of positional encodings (default: 5000).
        """
        super().__init__()

        idx = 1.0 / 10000 ** (torch.arange(0, input_size, 2) / input_size)
        pos = torch.arange(0, max_len).reshape(max_len, 1)

        self.embedding = torch.zeros((max_len, input_size))
        self.embedding[:, 0::2] = torch.sin(pos * idx)
        self.embedding[:, 1::2] = torch.cos(pos * idx)
        self.embedding = self.embedding.unsqueeze(0)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): Tensor of shape (B, T, C)
        Returns:
            x (Tensor): Tensor of shape (B, T, C)
        """
        return x + self.embedding[:, : x.shape[1]].to(x.device)


class MultiHeadedSelfAttention(nn.Module):
    def __init__(
        self,
        input_size: int,
        n_head: int = 8,
        p_drop: float = 0.1,
        max_len: int = 5000,
    ):
        """
        Args:
            input_size (int): input size.
            n_head (int): number of attention heads (default: 8).
            p_drop (float): probability of dropout (default: 0.1).
            max_len (int): max length of positional encodings (default: 5000).
        """
        super().__init__()

        self.preprocess = nn.Sequential(
            nn.LayerNorm(normalized_shape=input_size),
            PositionalEncoding(input_size, max_len=max_len),
        )
        self.mhsa = nn.MultiheadAttention(embed_dim=input_size, num_heads=n_head, batch_first=True)
        self.postprocess = nn.Sequential(
            nn.Dropout(p=p_drop),
            nn.LayerNorm(normalized_shape=input_size),
            nn.Dropout(p=p_drop),
        )

    def forward(self, x: Tensor, padding_mask: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            x (Tensor): Tensor of shape (B, T, C)
            padding_mask (Optional[Tensor]): padding mask to use in MHSA
        Returns:
            x (Tensor): Tensor of shape (B, T, C)
        """

        out = x.transpose(1, 2)
        out = self.preprocess(out)

        out, _ = self.mhsa(
            query=out,
            key=out,
            value=out,
            need_weights=False,
            key_padding_mask=padding_mask,
        )
        out = self.postprocess(out)

        out = out.transpose(1, 2)

        return out + x


class MHSA(nn.Module):
    """
    Multi head self attention
    """

    def __init__(
        self,
        input_size: int,
        n_head: int = 8,
    ):
        """
        Args:
            input_size (int): input size.
            n_head (int): number of attention heads (default: 8).
        """
        super(MHSA, self).__init__()

        self.mhsa = nn.Sequential(
            MultiHeadedSelfAttention(input_size=input_size, n_head=n_head), FeedForward(input_size=input_size)
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): input tensor (B, C, L).
        Return:
            output (Tensor): output tensor (B, C, L).
        """

        output = self.mhsa(x)

        return output + x

from typing import Optional

import torch
from rtfs_block import ChannelNorm
from torch import Tensor, nn


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
                    nn.Conv1d(
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
                    nn.Conv1d(
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
                    nn.Conv1d(
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
            nn.Conv1d(
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

        attention = torch.matmul(queries, keys.transpose(1, 2)) / torch.sqrt(queries.shape[3])  # (B * n_head, T, T)
        attention = nn.functional.softmax(attention, dim=-1)  # (B * n_head, T, T)
        output = torch.matmul(attention, values)  # (B * n_head, T, C / n_head * F)
        output = output.view(B * self.n_head, T, C // self.n_head, F)
        output = output.transpose(1, 2)  # (B * n_head, C / n_head, T, F)

        output = output.view(self.n_head, B, C // self.n_head, T, F)
        output = output.transpose(1, 2).contiguous()  # (B, n_head, C / n_head, T, F)
        output = output.view(B, C, T, F)

        output = self.proj(output)  # (B, C, T, F)
        output = output + x

        return output

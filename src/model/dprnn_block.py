import torch
from torch import Tensor, nn
from torch.nn import Sequential


class DPRNNBlock(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        """
        Args:
            input_size (int): input size
            hidden_size (int): hidden size
        """

        self.intra_rnn = nn.LSTM(
            input_size, hidden_size, batch_first=True, bidirectional=True
        )
        self.intra_fc = nn.Linear(hidden_size * 2, input_size)
        self.intra_ln = nn.GroupNorm(1, input_size)

        self.inter_rnn = nn.LSTM(
            input_size, hidden_size, batch_first=True, bidirectional=True
        )
        self.inter_fc = nn.Linear(hidden_size * 2, input_size)
        self.inter_ln = nn.GroupNorm(1, input_size)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): segmentation of x (B, C, chunk_size, L / chunk_size)
        """
        B, C, K, S = x.shape

        x = x.transpose(1, 3).contiguous().view(B * S, K, C)
        intra, _ = self.intra_rnn(x)  # (B * S, K, hidden_size * 2)
        intra = self.intra_fc(intra)  # (B * S, K, С)
        intra = intra.view(B, S, K, C).transpose(1, 3).contiguous()  # (B, C, K, S)
        intra = self.intra_ln(intra) + x  # (B, C, K, S)

        intra = intra.permute(0, 2, 3, 1).contiguous().view(B * K, S, C)
        inter, _ = self.inter_rnn(intra)  # (B * K, S, hidden_size * 2)
        inter = self.inter_fc(inter)  # (B * K, S, C)
        intra = intra.view(B, K, S, C).permute(0, 2, 3, 1).contiguous()  # (B, C, K, S)
        inter = self.inter_ln(inter) + x

        return inter


class DPRNN(nn.Module):
    """
    Implementation of the DUAL-PATH RNN

    https://arxiv.org/pdf/1910.06379
    """

    def __init__(
        self,
        input_size: int = 256,
        hidden_size: int = 128,
        num_speakers: int = 2,
        chunk_size: int = 250,
        num_dprnn_blocks: int = 6,
    ):
        """
        Args:
            input_size (int): input size
            hidden_size (int): hidden size
            num_speakers (int): number of speakers to separate
            chunk_size (int): chunk size (P in article)
            num_dprnn_blocks (int): number of DPRNN blocks (B in article)
        """
        assert chunk_size % 2 == 0, "Chunk size must be even"

        super().__init__()

        self.preprocess = nn.Sequential(
            nn.GroupNorm(1, input_size),
            nn.Conv1d(input_size, hidden_size, 1),
        )
        self.segmentation = nn.Unfold(
            kernel_size=(self.chunk_size, 1),
            padding=(self.chunk_size, 0),
            stride=(self.chunk_size // 2, 1),
        )
        self.model = nn.Sequential(
            *[DPRNNBlock(hidden_size, hidden_size) for _ in range(num_dprnn_blocks)],
            nn.PReLU(),
            nn.Conv2d(hidden_size, hidden_size * num_speakers, 1),  # speaker separation
        )
        self.mixture1 = nn.Sequential(
            nn.Conv1d(hidden_size, hidden_size, 1),
            nn.Sigmoid(),
        )
        self.mixture2 = nn.Sequential(
            nn.Conv1d(hidden_size, hidden_size, 1),
            nn.Tanh(),
        )
        self.outprocess = nn.Sequential(
            nn.Conv1d(hidden_size, input_size, 1),
            nn.ReLU(),
        )

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_speakers = num_speakers
        self.chunk_size = chunk_size
        self.num_dprnn_blocks = num_dprnn_blocks

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): input tensor of size (B, C, L)
        Outputs:
            output (Tensor): output tensor of size (B * num_speakers, C, L)
        """

        B, C, L = x.shape
        P = self.chunk_size // 2

        processed = self.preprocess(x)
        segmentation = self.segmentation(processed.unsqueeze(-1))
        output = self.model(segmentation)

        B, _, K, S = output.shape
        output = output.view(B * self.num_speakers, self.hidden_size * K, S)

        overlapped = nn.functional.fold(
            output,
            (L, 1),
            kernel_size=(self.chunk_size, 1),
            padding=(self.chunk_size, 0),
            stride=(P, 1),
        ).view(B * self.num_speakers, self.hidden_size, -1)

        output = self.mixture1(overlapped) * self.mixture2(overlapped)

        return self.outprocess(output)

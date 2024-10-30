import torch
from torch import nn


class BaselineModel(nn.Module):
    """
    Simple Encoder Decoder speech separator
    """

    def __init__(self, c_sources=2, seg_length=256, dim=128):
        """
        Args:
            c_sources (int): number of target sources.
            seg_length (int): length of the segment from input audio.
            dim (int): number of generated features from segment.
        """
        super().__init__()
        self.c_sources = c_sources

        self.encoder = nn.Conv1d(1, dim, kernel_size=seg_length, stride=seg_length // 2)
        self.generate_masks = nn.Conv1d(dim, dim * c_sources, kernel_size=1)
        self.act = nn.Softmax(dim=0)

        self.decoder = nn.ConvTranspose1d(dim, 1, kernel_size=seg_length, stride=seg_length // 2)

    def forward(self, mixture, **batch):
        """
        Args:
            mixture (Tensor): mixed audio (batch_size, time).
        Returns:
            output (dict): output dict containing predicted separated sources (batch_size, c_sources, time).
        """
        x = mixture.unsqueeze(1)
        x = self.encoder(x)  # [batch_size, dim, time]

        masks = self.generate_masks(x)  # [batch_size, dim * c_sources, time]
        masks = torch.chunk(masks, chunks=self.c_sources, dim=1)  # c_sources * [batch_size, dim, time]
        masks = self.act(torch.stack(masks, dim=0))  # [c_sources, batch_size, dim, time]

        decoded = [
            self.decoder(x * masks[i]).squeeze(1) for i in range(self.c_sources)
        ]  # c_sources * [batch_size, time]
        sources = torch.stack(decoded, dim=1)  # [batch_size, c_sources, time]
        return {"preds": sources}

    def __str__(self):
        """
        Model prints with the number of parameters.
        """
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum([p.numel() for p in self.parameters() if p.requires_grad])

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info

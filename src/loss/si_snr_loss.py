import torch
from torch import nn
from torchmetrics.audio import ScaleInvariantSignalNoiseRatio


class SiSNRLoss(nn.Module):
    """
    Calculate permutation invariant SI-SNR loss.
    """

    def __init__(self):
        super().__init__()
        self.si_snr = ScaleInvariantSignalNoiseRatio()

    def forward(self, preds: torch.Tensor, sources: torch.Tensor, **batch):
        """
        Args:
            preds (Tensor): model separated audio predictions (batch_size, c_sources, time).
            sources (Tensor): ground-truth separated audio (batch_size, c_sources, time).
        Returns:
            losses (dict): dict containing calculated loss functions.
        """
        sources_swap = sources[:, [1, 0]]
        max_si_snr = torch.max(
            self.si_snr(preds, sources),
            self.si_snr(preds, sources_swap),
        )

        return {"loss": -max_si_snr}

import torch
from torch import nn
from torchmetrics.audio import ScaleInvariantSignalNoiseRatio


class SiSNRLoss(nn.Module):
    """
    Calculate permutation invariant SI-SNR loss.
    """

    def __init__(self, permutation_invariant_training: bool = True):
        """
        Args:
            permutation_invariant_training (bool): count maximum of all possible permutations during training.
        """
        super().__init__()
        self.si_snr = ScaleInvariantSignalNoiseRatio()
        self.permutation_invariant_training = permutation_invariant_training

    def forward(self, preds: torch.Tensor, sources: torch.Tensor, **batch):
        """
        Args:
            preds (Tensor): model separated audio predictions (batch_size, c_sources, time).
            sources (Tensor): ground-truth separated audio (batch_size, c_sources, time).
        Returns:
            losses (dict): dict containing calculated loss functions.
        """
        if self.permutation_invariant_training:
            sources_swap = sources[:, [1, 0]]
            max_si_snr = torch.max(
                self.si_snr(preds, sources),
                self.si_snr(preds, sources_swap),
            )
        else:
            max_si_snr = self.si_snr(preds, sources)

        return {"loss": -max_si_snr}

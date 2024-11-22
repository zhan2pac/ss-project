import torch
from torch import Tensor, nn

from src.loss.si_snr_loss import SiSNRLoss
from src.loss.si_snr_mag_loss import SiSNRMagnitudeLoss


class MultiLoss(nn.Module):
    """
    Calculate Multi-Loss.
    """

    def __init__(
        self,
        perm_invariant: bool = True,
        clip_value: int = 30,
        alpha: float = 0.4,
        n_fft: int = 512,
        hop_length: int = 128,
    ):
        """
        Args:
            perm_invariant (bool): choose target metric from all possible source audio permutations.
            clip_value (int): metric clipping value, limits the influence of the best training prediction.
            alpha (float): weight of auxiliary part in multi-loss
            n_fft (int): size of FFT, creates n_fft // 2 + 1 bins.
            hop_length (int): length of hop between STFT windows.
        """
        super().__init__()
        self.alpha = alpha
        self.si_snr_loss = SiSNRLoss(perm_invariant, clip_value)
        self.si_snr_mag_loss = SiSNRMagnitudeLoss(perm_invariant, clip_value, n_fft, hop_length)

    def forward(self, preds: Tensor, preds_aux: list[Tensor], sources: Tensor, **batch):
        """
        Args:
            preds (Tensor): model separated audio predictions (batch_size, c_sources, time).
            preds_aux (list): model separated audio predictions (batch_size, c_sources, time).
            sources (Tensor): ground-truth separated audio (batch_size, c_sources, time).
        Returns:
            losses (dict): dict containing calculated loss functions.
        """
        loss_main = self.si_snr_loss(preds, sources)["loss"]

        intermediate_losses = [self.si_snr_mag_loss(preds_value, sources)["loss"] for preds_value in preds_aux]
        loss_aux = sum(intermediate_losses) / len(intermediate_losses)

        epoch = batch["epoch"]
        if epoch > 100:
            if (epoch - 1) % 5 == 0:
                self.alpha *= 0.8

        multi_loss = (1 - self.alpha) * loss_main + self.alpha * loss_aux
        return {"loss": multi_loss, "loss_main": loss_main, "loss_aux": loss_aux}

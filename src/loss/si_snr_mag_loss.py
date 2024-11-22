from itertools import permutations

import torch
import torchaudio
from torch import nn


class SiSNRMagnitudeLoss(nn.Module):
    """
    Calculate permutation invariant SI-SNR Magnitude loss.
    """

    def __init__(self, perm_invariant: bool = True, clip_value: int = 30, n_fft: int = 512, hop_length: int = 128):
        """
        Args:
            perm_invariant (bool): choose target metric from all possible source audio permutations.
            clip_value (int): metric clipping value, limits the influence of the best training prediction.
            n_fft (int): size of FFT, creates n_fft // 2 + 1 bins.
            hop_length (int): length of hop between STFT windows.
        """
        super().__init__()
        self.stft = torchaudio.transforms.Spectrogram(n_fft=n_fft, hop_length=hop_length)
        self.perm_invariant = perm_invariant
        self.clip_value = clip_value

    def si_snr_mag(self, preds, target):
        eps = torch.finfo(preds.dtype).eps

        target = target - torch.mean(target, dim=-1, keepdim=True)
        preds = preds - torch.mean(preds, dim=-1, keepdim=True)

        scale = (torch.sum(preds * target, dim=-1, keepdim=True) + eps) / (
            torch.sum(target**2, dim=-1, keepdim=True) + eps
        )
        target = target * torch.clamp(scale, min=1e-2)

        target_spec = self.stft(target)
        preds_spec = self.stft(preds)
        noise = target_spec - preds_spec

        val = (torch.sum(target_spec**2, dim=[-1, -2]) + eps) / (torch.sum(noise**2, dim=[-1, -2]) + eps)
        val = 10 * torch.log10(val)
        val = val.clamp(max=self.clip_value)

        return val.mean()

    def forward(self, preds: torch.Tensor, sources: torch.Tensor, **batch):
        """
        Args:
            preds (Tensor): model separated audio predictions (batch_size, c_sources, time).
            sources (Tensor): ground-truth separated audio (batch_size, c_sources, time).
        Returns:
            losses (dict): dict containing calculated loss functions.
        """
        _, c_sources, _ = preds.shape

        if self.perm_invariant:
            metrics_perm = []
            for permute in permutations(range(c_sources)):
                metric = self.si_snr_mag(preds, sources[:, permute])
                metrics_perm.append(metric)
            target_metric = max(metrics_perm)
        else:
            target_metric = self.si_snr_mag(preds, sources)

        return {"loss": -target_metric}

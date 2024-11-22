from itertools import permutations

import torch
from torchmetrics.audio import SignalDistortionRatio

from src.metrics.base_metric import BaseMetric


class SDRi(BaseMetric):
    def __init__(self, device: str, audio_only: bool, *args, **kwargs):
        """
        Use TorchMetrics SignalDistortionRatio function inside.

        Args:
            device (str): device for the metric calculation (and tensors).
            audio_only (bool): use permute technic to calculate metric.
        """
        super().__init__(*args, **kwargs)

        metric = SignalDistortionRatio()

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.metric = metric.to(device)
        self.audio_only = audio_only

    def __call__(self, mixture: torch.Tensor, preds: torch.Tensor, sources: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Calculate SDRi metric.

        Args:
            mixture (Tensor): input mixed audio (B, Time).
            preds (Tensor): model predictions (B, c_sources, Time).
            sources (Tensor): ground-truth audio (B, c_sources, Time).
        Returns:
            metrics (Tensor): calculated SDRi.
        """
        if self.audio_only:
            _, c_sources, _ = preds.shape
            mixture = mixture.unsqueeze(1).repeat(1, c_sources, 1)

            metrics_perm = []
            for permute in permutations(range(c_sources)):
                metric = self.metric(preds, sources[:, permute]) - self.metric(mixture, sources[:, permute])
                metrics_perm.append(metric)

            return max(metrics_perm)
        else:
            _, n_sources, _ = preds.shape

            metrics = 0
            for i in range(n_sources):
                metrics += self.metric(preds[:, i], sources[:, i]) - self.metric(mixture, sources[:, i])

            return metrics / n_sources

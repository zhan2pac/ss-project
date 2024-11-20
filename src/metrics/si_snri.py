from itertools import permutations

import torch
from torchmetrics.audio import ScaleInvariantSignalNoiseRatio

from src.metrics.base_metric import BaseMetric


class SiSNRi(BaseMetric):
    def __init__(self, device, *args, **kwargs):
        """
        Use TorchMetrics ScaleInvariantSignalNoiseRatio function inside.

        Args:
            device (str): device for the metric calculation (and tensors).
        """
        super().__init__(*args, **kwargs)

        metric = ScaleInvariantSignalNoiseRatio()

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.metric = metric.to(device)

    def __call__(self, mixture: torch.Tensor, preds: torch.Tensor, sources: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Calculate SI-SNRi metric.

        Args:
            mixture (Tensor): input mixed audio (B, Time).
            preds (Tensor): model predictions (B, c_sources, Time).
            sources (Tensor): ground-truth audio (B, c_sources, Time).
        Returns:
            metrics (Tensor): calculated SDRi.
        """
        _, c_sources, _ = preds.shape
        mixture = mixture.unsqueeze(1).repeat(1, c_sources, 1)

        metrics_perm = []
        for permute in permutations(range(c_sources)):
            metric = self.metric(preds, sources[:, permute]) - self.metric(mixture, sources[:, permute])
            metrics_perm.append(metric)

        return max(metrics_perm)

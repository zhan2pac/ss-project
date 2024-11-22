from itertools import permutations

import torch
from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility

from src.metrics.base_metric import BaseMetric


class STOI(BaseMetric):
    def __init__(self, fs: int, device: str, *args, **kwargs):
        """
        Use TorchMetrics ShortTimeObjectiveIntelligibility function inside.

        Args:
            fs (int): sampling frequency (Hz).
            device (str): device for the metric calculation (and tensors).
        """
        super().__init__(*args, **kwargs)

        metric = ShortTimeObjectiveIntelligibility(fs)

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.metric = metric.to(device)

    def __call__(self, preds: torch.Tensor, sources: torch.Tensor, **batch) -> torch.Tensor:
        """
        Calculate PESQ metric.

        Args:
            preds (Tensor): model predictions (..., Time).
            sources (Tensor): ground-truth audio (..., Time).
        Returns:
            metrics (Tensor): calculated STOI.
        """
        _, c_sources, _ = preds.shape

        metrics_perm = []
        for permute in permutations(range(c_sources)):
            metric = self.metric(preds, sources[:, permute])
            metrics_perm.append(metric)

        return max(metrics_perm)

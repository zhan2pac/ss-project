import torch
from torchmetrics.audio import SignalDistortionRatio

from src.metrics.base_metric import BaseMetric


class SDRi(BaseMetric):
    def __init__(self, device, *args, **kwargs):
        """
        Use TorchMetrics SignalDistortionRatio function inside.

        Args:
            device (str): device for the metric calculation (and tensors).
        """
        super().__init__(*args, **kwargs)

        metric = SignalDistortionRatio()

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.metric = metric.to(device)

    def __call__(self, mixture: torch.Tensor, preds: torch.Tensor, sources: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Calculate SDRi metric.

        Args:
            mixture (Tensor): input mixed audio (B, Time).
            preds (Tensor): model predictions (B, n_sources, Time).
            sources (Tensor): ground-truth audio (B, n_sources, Time).
        Returns:
            metrics (Tensor): calculated SDRi.
        """
        _, n_sources, _ = preds.shape

        metrics = 0
        for i in range(n_sources):
            metrics += self.metric(preds[:, i], sources[:, i]) - self.metric(mixture, sources[:, i])

        return metrics / n_sources

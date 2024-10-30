import torch
from torchmetrics.audio import ScaleInvariantSignalNoiseRatio

from src.metrics.base_metric import BaseMetric


class SiSNR(BaseMetric):
    def __init__(self, device, *args, **kwargs):
        """
        Use TorchMetrics ScaleInvariantSignalNoiseRatio function inside.

        Args:
            device (str): device for the metric calculation (and tensors).
        """
        super().__init__(*args, **kwargs)

        metric = ScaleInvariantSignalNoiseRatio(*args, **kwargs)

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.metric = metric.to(device)

    def __call__(self, preds: torch.Tensor, sources: torch.Tensor, **batch) -> torch.Tensor:
        """
        Calculate SI-SNR metric.

        Args:
            preds (Tensor): model predictions (..., Time).
            sources (Tensor): ground-truth audio (..., Time).
        Returns:
            metrics (Tensor): calculated SI-SNR.
        """

        return self.metric(preds, sources)

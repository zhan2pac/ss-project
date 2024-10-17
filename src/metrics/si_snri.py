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

        metric = ScaleInvariantSignalNoiseRatio(*args, **kwargs)

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.metric = metric.to(device)

    def __call__(
        self, inputs: torch.Tensor, preds: torch.Tensor, labels: torch.Tensor, **batch
    ) -> torch.Tensor:
        """
        Calculate SI-SNRi metric.

        Args:
            inputs (Tensor): input mixed audio (..., Time).
            preds (Tensor): model predictions (..., Time).
            labels (Tensor): ground-truth audio (..., Time).
        Returns:
            metrics (Tensor): calculated SI-SNRi.
        """

        return self.metric(preds, labels) - self.metric(inputs, labels)

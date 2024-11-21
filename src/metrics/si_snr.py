import torch
from torchmetrics.audio import ScaleInvariantSignalNoiseRatio

from src.metrics.base_metric import BaseMetric


class SiSNR(BaseMetric):
    def __init__(self, device: str, audio_only: bool, *args, **kwargs):
        """
        Use TorchMetrics ScaleInvariantSignalNoiseRatio function inside.

        Args:
            device (str): device for the metric calculation (and tensors).
            audio_only (bool): use permute technic to calculate metric.
        """
        super().__init__(*args, **kwargs)

        metric = ScaleInvariantSignalNoiseRatio()

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.metric = metric.to(device)
        self.audio_only = audio_only

    def __call__(self, preds: torch.Tensor, sources: torch.Tensor, **batch) -> torch.Tensor:
        """
        Calculate SI-SNR metric.

        Args:
            preds (Tensor): model predictions (..., Time).
            sources (Tensor): ground-truth audio (..., Time).
        Returns:
            metrics (Tensor): calculated SI-SNR.
        """

        if self.audio_only:
            sources_swap = sources[:, [1, 0]]
            metric_value = torch.max(
                self.metric(preds, sources),
                self.metric(preds, sources_swap),
            )
        else:
            metric_value = self.metric(preds, sources)

        return metric_value

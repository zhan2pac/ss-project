import torch
from torchmetrics.audio import ScaleInvariantSignalNoiseRatio

from src.metrics.base_metric import BaseMetric


class SiSNRi(BaseMetric):
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

    def __call__(self, mixture: torch.Tensor, preds: torch.Tensor, sources: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Calculate SI-SNRi metric.

        Args:
            mixture (Tensor): input mixed audio (B, Time).
            preds (Tensor): model predictions (B, n_sources, Time).
            sources (Tensor): ground-truth audio (B, n_sources, Time).
        Returns:
            metrics (Tensor): calculated SDRi.
        """
        _, n_sources, _ = preds.shape

        if self.audio_only:
            sources_swap = sources[:, [1, 0]]
            metrics = 0
            metrics_swap = 0
            for i in range(n_sources):
                metrics += self.metric(preds[:, i], sources[:, i]) - self.metric(mixture, sources[:, i])
                metrics_swap += self.metric(preds[:, i], sources_swap[:, i]) - self.metric(mixture, sources_swap[:, i])

            return torch.max(
                metrics / n_sources,
                metrics_swap / n_sources,
            )
        else:
            metrics = 0
            for i in range(n_sources):
                metrics += self.metric(preds[:, i], sources[:, i]) - self.metric(mixture, sources[:, i])

            return metrics / n_sources
